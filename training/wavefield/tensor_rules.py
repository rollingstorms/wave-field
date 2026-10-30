"""Batched sparse rules on CPU, MPS, or CUDA; Rust remains the reference engine.

Only the current full-strength product configuration is accepted. UI/history JSON
is converted at the boundary, never inside candidate move evaluation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import PIECE_IDS, PIECE_TYPES, PLAYERS, TUNING_SLOTS, ACTION_SIZE, TUNING_ACTION_SIZE
from .engine import load_initial_state


@dataclass
class TensorState:
    positions: torch.Tensor  # [batch, 12], -1 means removed
    unstable: torch.Tensor  # [batch, 12], saved rescue deadline flags
    components: torch.Tensor  # [batch, 2, 7], red then blue; absent spy slot stays zero
    order: torch.Tensor  # [batch, 2, 7], activation rank within type (repetition identity)
    player: torch.Tensor  # [batch], red=0, blue=1
    turn: torch.Tensor  # [batch], display turn counter
    status: torch.Tensor  # [batch], playing=0, red won=1, blue won=2

    def __len__(self):
        return self.positions.shape[0]

    def take(self, index):
        return TensorState(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    @classmethod
    def cat(cls, states):
        return cls(**{f.name: torch.cat([getattr(s, f.name) for s in states]) for f in fields(cls)})


class TensorRules:
    def __init__(self, device: str | torch.device = 'cpu', fused: bool = True):
        self.device = torch.device(device)
        initial = load_initial_state()
        self.template = initial
        by_id = {piece['id']: piece for piece in initial['pieces']}
        self.owners = self.tensor([PLAYERS.index(by_id[p]['owner']) for p in PIECE_IDS], torch.long)
        self.types = self.tensor([PIECE_TYPES.index(by_id[p]['type']) for p in PIECE_IDS], torch.long)
        self.signs = (1 - 2*self.owners).float()
        self.first_slots = self.tensor([0, 1, 3, 5], torch.long)
        self.second_slots = self.tensor([0, 2, 3, 6], torch.long)
        self.present_slots = self.tensor([1, 1, 1, 1, 0, 1, 1], torch.bool)
        self.tune_values = self.tensor([-1, 1]*7, torch.float32)
        self.tune_slots = torch.arange(14, device=self.device)//2
        # Integers represented exactly in float32; compatible with MPS mat/gather kernels.
        self.kernels = self.tensor(self._kernels(), torch.float32)
        rays=np.zeros((49,8,6),dtype=np.int64)
        valid=np.zeros_like(rays,dtype=bool)
        directions=[(dx,dy) for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy]
        for square in range(49):
            y,x=divmod(square,7)
            for direction,(dx,dy) in enumerate(directions):
                for distance in range(1,7):
                    nx,ny=x+dx*distance,y+dy*distance
                    if 0<=nx<7 and 0<=ny<7:
                        rays[square,direction,distance-1]=ny*7+nx
                        valid[square,direction,distance-1]=True
        self.rays=self.tensor(rays,torch.long)
        self.ray_valid=self.tensor(valid,torch.bool)
        self.metal=None
        if self.device.type=="mps" and fused and hasattr(torch.mps,"compile_shader"):
            from .metal_rules import MetalRules
            self.metal=MetalRules(self)

    def tensor(self, data, dtype):
        return torch.as_tensor(data, dtype=dtype, device=self.device)

    @staticmethod
    def _kernels():
        result=np.zeros((4,4,49,49),np.float32)
        for kind in range(4):
            for code in range(4):
                first=1 if code&1 else -1
                second=1 if code&2 else -1
                for source in range(49):
                    sy,sx=divmod(source,7)
                    for target in range(49):
                        ty,tx=divmod(target,7);x,y=tx-sx,ty-sy
                        radius=max(abs(x),abs(y))
                        axis=(x==0)!=(y==0)
                        diagonal=x!=0 and abs(x)==abs(y)
                        diamond=abs(x)+abs(y)<=2
                        value=0
                        if x==0 and y==0:
                            value=int(kind==2)
                        elif radius<=2:
                            if kind==0:
                                value=int(axis) if first==1 else 2*int(diagonal and radius==1)
                            elif kind==2:
                                value=int(diamond) if first==1 else 3*int(axis and radius==1)
                            elif first==1 and second==1:
                                value=int(kind==3 or axis or diagonal)
                            elif first==1 and second==-1:
                                value=2*int(axis if kind==1 else diamond)
                            elif first==-1 and second==1:
                                value=(3 if kind==3 else 2)*int(diagonal)
                            else:
                                value=(6 if kind==3 else 4)*int(axis and radius==1)
                        result[kind,code,source,target]=value
        return result

    def pack(self, states: Sequence[dict[str, Any]]) -> TensorState:
        positions=[];unstable=[];components=[];orders=[]
        for state in states:
            if state.get('variant')!='sparse':
                raise ValueError('Tensor rules support the sparse variant only')
            by_id={p['id']:p for p in state['pieces']}
            if len(by_id)!=len(state['pieces']) or not set(by_id).issubset(PIECE_IDS):
                raise ValueError('Tensor rules require unique product piece identities')
            occupied=set()
            for piece in by_id.values():
                owner,kind,_=piece['id'].split('-')
                x,y=piece['position']['x'],piece['position']['y']
                if piece['owner']!=owner or piece['type']!=kind or not (0<=x<7 and 0<=y<7) or (x,y) in occupied:
                    raise ValueError('Invalid product piece placement or identity')
                occupied.add((x,y))
            positions.append([by_id[p]['position']['y']*7+by_id[p]['position']['x'] if p in by_id else -1 for p in PIECE_IDS])
            unstable.append([bool(by_id[p].get('unstable',False)) if p in by_id else False for p in PIECE_IDS])
            comp=[];order=[]
            for player in PLAYERS:
                expected={'pawn':1,'rook':2,'spy':1,'king':2}
                for kind,count in expected.items():
                    values=state['components'][player][kind]
                    if sorted(state['activationOrders'][player][kind])!=list(range(count)):
                        raise ValueError('Tensor rules require a complete activation order')
                    if len(values)!=count or any(value not in (-1,1) for value in values):
                        raise ValueError('Tensor rules require current full-strength product controls')
                comp.append([state['components'][player][kind][slot] if slot<len(state['components'][player][kind]) else 0
                             for kind,slot in TUNING_SLOTS])
                order.append([state['activationOrders'][player][kind].index(slot) if slot in state['activationOrders'][player][kind] else -1
                              for kind,slot in TUNING_SLOTS])
            components.append(comp);orders.append(order)
        return TensorState(self.tensor(positions,torch.long),self.tensor(unstable,torch.bool),
                           self.tensor(components,torch.float32),self.tensor(orders,torch.long),
                           self.tensor([PLAYERS.index(s['currentPlayer']) for s in states],torch.long),
                           self.tensor([s['turnNumber'] for s in states],torch.long),
                           self.tensor([('playing','red-won','blue-won').index(s['status']) for s in states],torch.long))

    def unpack(self, state: TensorState, templates=None) -> list[dict[str, Any]]:
        arrays={f.name:getattr(state,f.name).detach().cpu().numpy() for f in fields(state)}
        result=[]
        for row in range(len(state)):
            output=copy.deepcopy(templates[row] if templates is not None else self.template)
            output.setdefault('ampSquares',[])
            # Match Rust's JSON schema at the display/repetition boundary, where
            # integer literals in fixtures become f64 fields after a transition.
            def normalize_definition(value):
                if isinstance(value,dict):
                    for key,item in value.items():
                        if key in ('decayBase','originScale','weight'):
                            value[key]=float(item)
                        else:normalize_definition(item)
                elif isinstance(value,list):
                    for item in value:normalize_definition(item)
            normalize_definition(output['definitions'])
            for scale in output['waveScales'].values():
                for key in scale:scale[key]=float(scale[key])
            output['homeEnergy']={key:float(value) for key,value in output['homeEnergy'].items()}
            by_id={p['id']:p for p in output['pieces']}
            pieces=[]
            for slot,piece_id in enumerate(PIECE_IDS):
                square=int(arrays['positions'][row,slot])
                if square<0:continue
                piece=by_id[piece_id]
                piece.update(position=dict(x=square%7,y=square//7),unstable=bool(arrays['unstable'][row,slot]))
                pieces.append(piece)
            output['pieces']=pieces
            for owner,player in enumerate(PLAYERS):
                for kind in PIECE_TYPES:
                    slots=[i for i,(k,j) in enumerate(TUNING_SLOTS) if k==kind and not (kind=='spy' and j==1)]
                    output['components'][player][kind]=[int(arrays['components'][row,owner,i]) for i in slots]
                    output['activationOrders'][player][kind]=sorted(range(len(slots)),key=lambda j:arrays['order'][row,owner,slots[j]])
            output.update(currentPlayer=PLAYERS[int(arrays['player'][row])],turnNumber=int(arrays['turn'][row]),
                          status=('playing','red-won','blue-won')[int(arrays['status'][row])],
                          selectedPieceId=None,history=[],message='')
            result.append(output)
        return result

    def contributions(self, state: TensorState):
        coefficients=state.components[:,self.owners,:]
        first=coefficients.gather(2,self.first_slots[self.types][None,:,None].expand(len(state),-1,1)).squeeze(-1)
        second=coefficients.gather(2,self.second_slots[self.types][None,:,None].expand(len(state),-1,1)).squeeze(-1)
        code=(first>0).long()+2*(second>0).long()
        return self.kernels[self.types[None,:],code,state.positions.clamp_min(0)] * self.signs[None,:,None] * (state.positions>=0)[:,:,None]

    def field(self, state: TensorState):
        if self.metal is not None:return self.metal.field(state)
        return self.contributions(state).sum(1)

    def instability(self, positions, field):
        values=field.gather(1,positions.clamp_min(0))
        return (positions>=0) & (values*self.signs[None,:]<0)

    def _resolve(self, positions, contributions, deadline):
        """Cascade only the mover's previously unstable non-kings, exactly as Rust."""
        contributions=contributions.clone()
        positions=positions.clone()
        # At most five eligible own pieces. CPU can finish early; GPU uses fixed
        # device-side passes to avoid synchronizing to Python at every cascade.
        for _ in range(5):
            field=contributions.sum(1)
            lost=self.instability(positions,field)&deadline
            if self.device.type=='cpu' and not lost.any():break
            positions=positions.masked_fill(lost,-1)
            contributions=contributions.masked_fill(lost[:,:,None],0)
        field=contributions.sum(1)
        unstable=self.instability(positions,field)
        return positions,unstable,field

    def raw_moves(self, state: TensorState, field=None):
        field=self.field(state) if field is None else field
        positions=state.positions
        occupancy=F.one_hot(positions.clamp_min(0),49).bool() & (positions>=0)[:,:,None]
        occupied=occupancy.any(1)
        squares=self.rays[positions.clamp_min(0)]
        valid=self.ray_valid[positions.clamp_min(0)]
        values=field.gather(1,squares.reshape(len(state),-1)).reshape_as(squares)
        blocked=occupied.gather(1,squares.reshape(len(state),-1)).reshape_as(squares)
        passable=(values*self.signs[None,:,None,None]>=0)|(self.types[None,:,None,None]==2)
        steps=valid&~blocked&passable
        reachable=steps.int().cumprod(-1).bool()
        # A scatter sum avoids invalid padded ray entries overwriting square 0.
        mask=torch.zeros((len(state),12,49),dtype=torch.float32,device=self.device)
        mask.scatter_add_(2,squares.flatten(2),reachable.flatten(2).float())
        own=(self.owners[None,:]==state.player[:,None])&(positions>=0)&(state.status[:,None]==0)
        return (mask>0)&own[:,:,None]

    def legal_masks(self, state: TensorState):
        if self.metal is not None:return self.metal.masks(state)
        return self.legal_moves(state),self.tuning_mask(state)

    def legal_moves(self, state: TensorState):
        if self.metal is not None:return self.metal.masks(state)[0]
        contribution=self.contributions(state)
        field=contribution.sum(1)
        raw=self.raw_moves(state,field)
        parents,pieces,destinations=raw.nonzero(as_tuple=True)
        mask=torch.zeros((len(state),12,49),dtype=torch.bool,device=self.device)
        if parents.numel()==0:return mask.reshape(len(state),ACTION_SIZE)
        positions=state.positions[parents].clone()
        candidate_contribution=contribution[parents].clone()
        coefficients=state.components[parents,self.owners[pieces]]
        kinds=self.types[pieces]
        first=coefficients.gather(1,self.first_slots[kinds,None]).squeeze(1)
        second=coefficients.gather(1,self.second_slots[kinds,None]).squeeze(1)
        code=(first>0).long()+2*(second>0).long()
        rows=torch.arange(parents.numel(),device=self.device)
        positions[rows,pieces]=destinations
        candidate_contribution[rows,pieces]=self.kernels[kinds,code,destinations]*self.signs[pieces,None]
        deadline=(state.unstable|self.instability(state.positions,field))
        deadline &= (self.owners[None,:]==state.player[:,None]) & (self.types[None,:]!=3)
        positions,unstable,_=self._resolve(positions,candidate_contribution,deadline[parents])
        unsafe=(unstable&(self.types[None,:]==3)&(self.owners[None,:]==state.player[parents,None])).any(1)
        mask[parents,pieces,destinations]=~unsafe
        return mask.reshape(len(state),ACTION_SIZE)

    def tuning_mask(self, state: TensorState):
        coefficients=state.components[torch.arange(len(state),device=self.device),state.player]
        return (coefficients[:,self.tune_slots]!=self.tune_values[None,:])&self.present_slots[self.tune_slots][None,:]&(state.status[:,None]==0)

    def step(self, state: TensorState, actions: torch.Tensor) -> TensorState:
        """Apply already-validated actions (0..587 moves, 588..601 tunes).

        No-move terminal detection belongs to search after computing next masks;
        this matches Rust applyMove(analyzeCheckmate=false).
        """
        if self.metal is not None:return self.metal.step(state,actions)
        batch=len(state);rows=torch.arange(batch,device=self.device)
        tune=actions>=ACTION_SIZE
        piece=(actions//49).clamp_max(11)
        destination=actions%49
        tune_index=(actions-ACTION_SIZE).clamp(0,TUNING_ACTION_SIZE-1)
        slot=tune_index//2
        values=self.tune_values[tune_index]
        positions=state.positions.clone()
        positions[rows,piece]=torch.where(tune,positions[rows,piece],destination)
        components=state.components.clone()
        components[rows,state.player,slot]=torch.where(tune,values,components[rows,state.player,slot])
        order=state.order.clone()
        # Reordering full-strength two-component controls requires no evictions.
        paired=torch.where(slot==1,2,torch.where(slot==2,1,torch.where(slot==5,6,5)))
        double=(slot==1)|(slot==2)|(slot==5)|(slot==6)
        order[rows,state.player,paired]=torch.where(tune&double,0,order[rows,state.player,paired])
        order[rows,state.player,slot]=torch.where(tune,double.long(),order[rows,state.player,slot])
        candidate=TensorState(positions,state.unstable,components,order,state.player,state.turn,state.status)
        old_field=self.field(state)
        deadline=(state.unstable|self.instability(state.positions,old_field))
        deadline &= (self.owners[None,:]==state.player[:,None])&(self.types[None,:]!=3)&~tune[:,None]
        positions,unstable,_=self._resolve(positions,self.contributions(candidate),deadline)
        return TensorState(positions,unstable,components,order,
                           torch.where(tune,state.player,1-state.player),
                           state.turn+((~tune)&(state.player==0)).long(),state.status.clone())

    def encode(self, state: TensorState, rich: bool = True):
        if self.metal is not None:return self.metal.encode(state,rich)
        alive=state.positions>=0
        occupancy=F.one_hot(state.positions.clamp_min(0),49).float()*alive[:,:,None]
        board=torch.zeros((len(state),25 if rich else 13,49),device=self.device)
        channels=self.owners*4+self.types
        board.scatter_add_(1,channels[None,:,None].expand(len(state),-1,49),occupancy)
        for owner in (0,1):
            board[:,8+owner]=(occupancy*(state.unstable&(self.owners[None,:]==owner))[:,:,None]).sum(1)
        field=self.field(state)
        board[:,10]=(field/8).clamp(-1,1)
        board[:,11]=(field.abs()/8).clamp(0,1)
        board[:,12]=(1-2*state.player).float()[:,None]
        if rich:board[:,13:25]=occupancy
        side=torch.cat((state.components.flatten(1),(1-2*state.player).float()[:,None]),1)
        return board.reshape(len(state),-1,7,7),side

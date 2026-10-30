"""Fused sparse-rule kernels for Apple's GPU, compiled through PyTorch MPS.

int2 accesses the low/high words of int64 state tensors; board coordinates and
turn counters fit int32. Portable Torch rules remain the CPU/CUDA reference.
"""
from functools import lru_cache
import torch

SOURCE = r'''
#include <metal_stdlib>
using namespace metal;
constant int owners[12] = {1,1,1,1,1,1,0,0,0,0,0,0};
constant int kinds[12] = {1,3,1,0,2,0,0,2,0,1,3,1};
constant int first_slot[4] = {0,1,3,5};
constant int second_slot[4] = {0,2,3,6};
inline int2 wide(int value) { return int2(value,value<0 ? -1 : 0); }
inline int coefficient_code(device const float* comp, int owner, int kind) {
    return int(comp[owner*7+first_slot[kind]]>0) + 2*int(comp[owner*7+second_slot[kind]]>0);
}
inline float contribution(device const float* kernels, int kind, int code, int source, int target) {
    return kernels[((kind*4+code)*49+source)*49+target];
}
inline float point_field(thread const int* pos, thread const int* codes,
                         device const float* kernels, int target) {
    float value=0;
    for (int p=0;p<12;p++) if(pos[p]>=0)
        value += float(1-2*owners[p])*contribution(kernels,kinds[p],codes[p],pos[p],target);
    return value;
}
inline void load_position(device const int2* positions, device const float* components,
                          thread int* pos, thread int* codes) {
    for(int p=0;p<12;p++) {
        pos[p]=positions[p].x;
        codes[p]=coefficient_code(components,owners[p],kinds[p]);
    }
}
inline void resolve(thread int* pos, thread const int* codes, thread const bool* deadline,
                    device const float* kernels) {
    for(int pass=0;pass<5;pass++) {
        uint lost=0;
        for(int p=0;p<12;p++) if(pos[p]>=0 && deadline[p]) {
            if(point_field(pos,codes,kernels,pos[p])*float(1-2*owners[p])<0) lost|=1u<<p;
        }
        if(!lost) break;
        for(int p=0;p<12;p++) if(lost&(1u<<p)) pos[p]=-1;
    }
}
kernel void sparse_field(device const int2* positions, device const float* components,
                         device const float* kernels, device float* field,
                         uint index [[thread_position_in_grid]]) {
    int row=index/49, target=index%49;
    int pos[12],codes[12];
    load_position(positions+row*12,components+row*14,pos,codes);
    field[index]=point_field(pos,codes,kernels,target);
}
kernel void sparse_masks(device const int2* positions, device const bool* unstable,
                         device const float* components, device const int2* players,
                         device const int2* statuses, device const float* kernels,
                         device const float* field, device bool* moves, device bool* tunes,
                         uint index [[thread_position_in_grid]]) {
    int row=index/602, action=index%602, player=players[row].x;
    if(action>=588) {
        int tune=action-588,slot=tune/2;
        float value=(tune%2) ? 1.0 : -1.0;
        tunes[row*14+tune]=statuses[row].x==0 && slot!=4 && components[row*14+player*7+slot]!=value;
        return;
    }
    moves[row*588+action]=false;
    int p=action/49,target=action%49,source=positions[row*12+p].x;
    if(statuses[row].x!=0 || owners[p]!=player || source<0 || target==source) return;
    int dx=target%7-source%7,dy=target/7-source/7;
    if(dx!=0 && dy!=0 && abs(dx)!=abs(dy)) return;
    int distance=max(abs(dx),abs(dy));
    int sx=(dx>0)-(dx<0),sy=(dy>0)-(dy<0);
    for(int step=1;step<=distance;step++) {
        int square=(source/7+sy*step)*7+source%7+sx*step;
        for(int other=0;other<12;other++) if(positions[row*12+other].x==square) return;
        if(kinds[p]!=2 && field[row*49+square]*float(1-2*player)<0) return;
    }
    int pos[12],codes[12];bool deadline[12];
    load_position(positions+row*12,components+row*14,pos,codes);
    for(int other=0;other<12;other++) {
        deadline[other]=pos[other]>=0 && owners[other]==player && kinds[other]!=3 &&
            (unstable[row*12+other] || field[row*49+pos[other]]*float(1-2*player)<0);
    }
    pos[p]=target;
    resolve(pos,codes,deadline,kernels);
    for(int other=0;other<12;other++) if(pos[other]>=0 && owners[other]==player && kinds[other]==3) {
        if(point_field(pos,codes,kernels,pos[other])*float(1-2*player)<0) return;
    }
    moves[row*588+action]=true;
}
kernel void sparse_step(device const int2* positions, device const bool* unstable,
                        device const float* components, device const int2* orders,
                        device const int2* players, device const int2* turns,
                        device const int2* statuses, device const int2* actions,
                        device const float* kernels,
                        device int2* out_pos, device bool* out_unstable,
                        device float* out_comp, device int2* out_order,
                        device int2* out_player,device int2* out_turn,device int2* out_status,
                        uint row [[thread_position_in_grid]]) {
    int action=actions[row].x, player=players[row].x;
    bool tune=action>=588;
    int pos[12],codes[12];bool deadline[12];
    load_position(positions+row*12,components+row*14,pos,codes);
    for(int p=0;p<12;p++) {
        deadline[p]=!tune && pos[p]>=0 && owners[p]==player && kinds[p]!=3 &&
            (unstable[row*12+p] || point_field(pos,codes,kernels,pos[p])*float(1-2*player)<0);
    }
    for(int i=0;i<14;i++) {
        out_comp[row*14+i]=components[row*14+i];
        out_order[row*14+i]=orders[row*14+i];
    }
    if(tune) {
        int t=action-588,slot=t/2;
        out_comp[row*14+player*7+slot]=(t%2) ? 1 : -1;
        bool paired=slot==1 || slot==2 || slot==5 || slot==6;
        if(paired) {
            int other=(slot==1) ? 2 : (slot==2) ? 1 : (slot==5) ? 6 : 5;
            out_order[row*14+player*7+other]=wide(0);
        }
        out_order[row*14+player*7+slot]=wide(int(paired));
        for(int p=0;p<12;p++) codes[p]=coefficient_code(out_comp+row*14,owners[p],kinds[p]);
    } else pos[action/49]=action%49;
    resolve(pos,codes,deadline,kernels);
    for(int p=0;p<12;p++) {
        out_pos[row*12+p]=wide(pos[p]);
        out_unstable[row*12+p]=pos[p]>=0 && point_field(pos,codes,kernels,pos[p])*float(1-2*owners[p])<0;
    }
    out_player[row]=wide(tune ? player : 1-player);
    out_turn[row]=wide(turns[row].x+int(!tune && player==0));
    out_status[row]=statuses[row];
}
kernel void sparse_encode(device const int2* positions,device const bool* unstable,
                          device const float* components,device const int2* players,
                          device const float* field,device float* board,device float* side,
                          constant int& channels,uint index [[thread_position_in_grid]]) {
    int row=index/(channels*49),channel=(index/49)%channels,square=index%49;
    float value=0;
    if(channel<8) {
        for(int p=0;p<12;p++) if(positions[row*12+p].x==square && owners[p]*4+kinds[p]==channel) value+=1;
    } else if(channel<10) {
        for(int p=0;p<12;p++) if(positions[row*12+p].x==square && owners[p]==channel-8 && unstable[row*12+p]) value+=1;
    } else if(channel==10) value=clamp(field[row*49+square]/8.0f,-1.0f,1.0f);
    else if(channel==11) value=clamp(abs(field[row*49+square])/8.0f,0.0f,1.0f);
    else if(channel==12) value=float(1-2*players[row].x);
    else value=float(positions[row*12+channel-13].x==square);
    board[index]=value;
    // The first 15 threads for each row also produce the side vector.
    if(channel==0 && square<15) side[row*15+square]=(square<14) ? components[row*14+square] : float(1-2*players[row].x);
}
'''


@lru_cache(maxsize=1)
def library():
    return torch.mps.compile_shader(SOURCE)


class MetalRules:
    def __init__(self, rules):
        self.rules=rules;self.lib=library()

    def field(self,state):
        result=torch.empty((len(state),49),device='mps')
        self.lib.sparse_field(state.positions,state.components,self.rules.kernels,result,threads=len(state)*49)
        return result

    def masks(self,state):
        moves=torch.empty((len(state),588),dtype=torch.bool,device='mps')
        tunes=torch.empty((len(state),14),dtype=torch.bool,device='mps')
        self.lib.sparse_masks(state.positions,state.unstable,state.components,state.player,state.status,
                              self.rules.kernels,self.field(state),moves,tunes,threads=len(state)*602)
        return moves,tunes

    def step(self,state,actions):
        from .tensor_rules import TensorState
        result=TensorState(**{name:torch.empty_like(getattr(state,name)) for name in state.__dataclass_fields__})
        self.lib.sparse_step(state.positions,state.unstable,state.components,state.order,state.player,
                             state.turn,state.status,actions,self.rules.kernels,result.positions,result.unstable,
                             result.components,result.order,result.player,result.turn,result.status,threads=len(state))
        return result

    def encode(self,state,rich):
        channels=25 if rich else 13
        board=torch.empty((len(state),channels,7,7),device='mps')
        side=torch.empty((len(state),15),device='mps')
        self.lib.sparse_encode(state.positions,state.unstable,state.components,state.player,self.field(state),
                               board,side,channels,threads=len(state)*channels*49)
        return board,side

struct Params { width: u32, height: u32 }
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn idx(x:u32,y:u32)->u32{return y*params.width+x;}

@compute @workgroup_size(16,16)
fn naive(@builtin(global_invocation_id) id:vec3<u32>){
  let x=id.x; let y=id.y;
  if(x>=params.width||y>=params.height){return;}
  if(x==0u||y==0u||x+1u>=params.width||y+1u>=params.height){output_data[idx(x,y)]=input_data[idx(x,y)];return;}
  var s=0.0;
  for(var dy:i32=-1;dy<=1;dy++){for(var dx:i32=-1;dx<=1;dx++){s+=input_data[idx(u32(i32(x)+dx),u32(i32(y)+dy))];}}
  output_data[idx(x,y)]=s/9.0;
}

var<workgroup> tile:array<f32,18*18>;
fn tid(x:u32,y:u32)->u32{return y*18u+x;}

@compute @workgroup_size(16,16)
fn tiled(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(global_invocation_id) id:vec3<u32>){
  let x=id.x;let y=id.y;let lx=lid.x+1u;let ly=lid.y+1u;
  if(x<params.width&&y<params.height){tile[tid(lx,ly)]=input_data[idx(x,y)];}
  if(lid.x==0u&&x>0u&&y<params.height){tile[tid(0u,ly)]=input_data[idx(x-1u,y)];}
  if(lid.x==15u&&x+1u<params.width&&y<params.height){tile[tid(17u,ly)]=input_data[idx(x+1u,y)];}
  if(lid.y==0u&&y>0u&&x<params.width){tile[tid(lx,0u)]=input_data[idx(x,y-1u)];}
  if(lid.y==15u&&y+1u<params.height&&x<params.width){tile[tid(lx,17u)]=input_data[idx(x,y+1u)];}
  if(lid.x==0u&&lid.y==0u&&x>0u&&y>0u){tile[tid(0u,0u)]=input_data[idx(x-1u,y-1u)];}
  if(lid.x==15u&&lid.y==0u&&x+1u<params.width&&y>0u){tile[tid(17u,0u)]=input_data[idx(x+1u,y-1u)];}
  if(lid.x==0u&&lid.y==15u&&x>0u&&y+1u<params.height){tile[tid(0u,17u)]=input_data[idx(x-1u,y+1u)];}
  if(lid.x==15u&&lid.y==15u&&x+1u<params.width&&y+1u<params.height){tile[tid(17u,17u)]=input_data[idx(x+1u,y+1u)];}
  workgroupBarrier();
  if(x>=params.width||y>=params.height){return;}
  if(x==0u||y==0u||x+1u>=params.width||y+1u>=params.height){output_data[idx(x,y)]=input_data[idx(x,y)];return;}
  var s=0.0;
  for(var dy:u32=0u;dy<3u;dy++){for(var dx:u32=0u;dx<3u;dx++){s+=tile[tid(lx+dx-1u,ly+dy-1u)];}}
  output_data[idx(x,y)]=s/9.0;
}

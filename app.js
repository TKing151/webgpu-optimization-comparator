const $=id=>document.getElementById(id);
let device,shader;

async function init(){
  try{
    if(!navigator.gpu)throw Error('WebGPU is not available in this browser.');
    const adapter=await navigator.gpu.requestAdapter();
    if(!adapter)throw Error('No WebGPU adapter was found.');
    device=await adapter.requestDevice();
    shader=await (await fetch('stencil.wgsl')).text();
    const info=adapter.info||{};
    $('device').textContent=[info.vendor,info.architecture,info.description].filter(Boolean).join(' / ')||'WebGPU adapter available';
    $('status').textContent='Ready';$('run').disabled=false;
  }catch(e){$('status').textContent='Unavailable';showError(e.message)}
}

function showError(s){$('error').textContent=s;$('error').classList.remove('hidden')}
function median(a){a=[...a].sort((x,y)=>x-y);return a.length%2?a[(a.length-1)/2]:(a[a.length/2-1]+a[a.length/2])/2}
function p95(a){a=[...a].sort((x,y)=>x-y);return a[Math.min(a.length-1,Math.floor(a.length*.95))]}
function inputData(n){const a=new Float32Array(n*n);for(let i=0;i<a.length;i++)a[i]=Math.sin(i*.001)*.5+Math.cos(i*.00037)*.25;return a}
function cpuRef(a,n){const o=new Float32Array(a.length);for(let y=0;y<n;y++)for(let x=0;x<n;x++){let i=y*n+x;if(x===0||y===0||x===n-1||y===n-1){o[i]=a[i];continue}let s=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++)s+=a[(y+dy)*n+x+dx];o[i]=s/9}return o}
function valid(a,b){let m=0;for(let i=0;i<a.length;i++)m=Math.max(m,Math.abs(a[i]-b[i]));return m<1e-4?{ok:true,max:m}:{ok:false,max:m}}

async function runKernel(n,input,entry,warmups,runs){
  const bytes=input.byteLength;
  const ib=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const ob=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const rb=device.createBuffer({size:bytes,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const pb=device.createBuffer({size:8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(ib,0,input);device.queue.writeBuffer(pb,0,new Uint32Array([n,n]));
  const module=device.createShaderModule({code:shader});
  const pipeline=device.createComputePipeline({layout:'auto',compute:{module,entryPoint:entry}});
  const bg=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:ib}},{binding:1,resource:{buffer:ob}},{binding:2,resource:{buffer:pb}}]});
  const dispatch=()=>{const e=device.createCommandEncoder();const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,bg);p.dispatchWorkgroups(Math.ceil(n/16),Math.ceil(n/16));p.end();device.queue.submit([e.finish()])};
  for(let i=0;i<warmups;i++){dispatch();await device.queue.onSubmittedWorkDone()}
  const samples=[];
  for(let i=0;i<runs;i++){const t=performance.now();dispatch();await device.queue.onSubmittedWorkDone();samples.push(performance.now()-t);$('progress').textContent=`${entry}: ${i+1}/${runs}`}
  const e=device.createCommandEncoder();e.copyBufferToBuffer(ob,0,rb,0,bytes);device.queue.submit([e.finish()]);await device.queue.onSubmittedWorkDone();await rb.mapAsync(GPUMapMode.READ);const out=new Float32Array(rb.getMappedRange().slice(0));rb.unmap();
  ib.destroy();ob.destroy();rb.destroy();pb.destroy();
  return {median:median(samples),p95:p95(samples),min:Math.min(...samples),max:Math.max(...samples),out}
}

async function runKernel(n,input,entry,warmups,runs){
  const bytes=input.byteLength;

  const ib=device.createBuffer({
    size:bytes,
    usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST
  });

  const ob=device.createBuffer({
    size:bytes,
    usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC
  });

  const rb=device.createBuffer({
    size:bytes,
    usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST
  });

  const pb=device.createBuffer({
    size:8,
    usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST
  });

  device.queue.writeBuffer(ib,0,input);
  device.queue.writeBuffer(pb,0,new Uint32Array([n,n]));

  const module=device.createShaderModule({code:shader});

  const pipeline=device.createComputePipeline({
    layout:'auto',
    compute:{module,entryPoint:entry}
  });

  const bg=device.createBindGroup({
    layout:pipeline.getBindGroupLayout(0),
    entries:[
      {binding:0,resource:{buffer:ib}},
      {binding:1,resource:{buffer:ob}},
      {binding:2,resource:{buffer:pb}}
    ]
  });

  const dispatch=()=>{
    const e=device.createCommandEncoder();
    const p=e.beginComputePass();

    p.setPipeline(pipeline);
    p.setBindGroup(0,bg);
    p.dispatchWorkgroups(
      Math.ceil(n/16),
      Math.ceil(n/16)
    );

    p.end();
    device.queue.submit([e.finish()]);
  };

  // Warmup
  for(let i=0;i<warmups;i++){
    dispatch();
    await device.queue.onSubmittedWorkDone();
  }

  // Timed runs
  const samples=[];

  for(let i=0;i<runs;i++){
    const t=performance.now();

    dispatch();

    await device.queue.onSubmittedWorkDone();

    const elapsed=performance.now()-t;
    samples.push(elapsed);

    $('progress').textContent=
      `${entry}: ${i+1}/${runs}`;
  }

  // Read result
  const e=device.createCommandEncoder();

  e.copyBufferToBuffer(
    ob,
    0,
    rb,
    0,
    bytes
  );

  device.queue.submit([e.finish()]);

  await device.queue.onSubmittedWorkDone();

  await rb.mapAsync(GPUMapMode.READ);

  const out=new Float32Array(
    rb.getMappedRange().slice(0)
  );

  rb.unmap();

  ib.destroy();
  ob.destroy();
  rb.destroy();
  pb.destroy();

  return {
    median:median(samples),
    p95:p95(samples),
    min:Math.min(...samples),
    max:Math.max(...samples),
    out
  };
}
init();

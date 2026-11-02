// No npm dependencies. Exercise extension callbacks with a minimal ComfyUI/DOM stub.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
let extension;
const events = new Map();
let node;
globalThis.testApp = {registerExtension: (e) => {extension=e;},graph:{getNodeById:(id)=>String(id)==='78'?node:null}};
globalThis.testApi = {addEventListener:(name,fn)=>events.set(name,fn)};
const fakeContext = {
    clearRect(){}, fillRect(){}, fillText(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}
};
globalThis.document = {createElement:()=>({
    style:{},setAttribute(){},value:'',readOnly:false,width:0,height:0,
    getContext:()=>fakeContext,
})};
const source = (await fs.readFile(new URL('../web/tiling_controls.js',import.meta.url),'utf8'))
  .replace('import { app } from "../../scripts/app.js";', 'const app=globalThis.testApp;')
  .replace('import { api } from "../../scripts/api.js";', 'const api=globalThis.testApi;');
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
class Node {
    constructor(){this.size=[300,300];this.widgets=[
        {name:'highres_tiling',value:true}, {name:'tiling_mode',value:'auto'},
        {name:'tiling_tiles',value:2}, {name:'tiling_axis',value:'auto'},
        {name:'highres_tiling_mode',value:'spatial'}, {name:'highres_token_chunk',value:4096},
        {name:'highres_weight_cache_gb',value:4}];}
    addDOMWidget(name,type,el,options){this.widgets.push({name,type,options});return this.widgets.at(-1);}
    computeSize(){return [300,300];}
    setDirtyCanvas(){}
}
await extension.beforeRegisterNodeDef(Node,{name:'SelfLiftAvatarH3Sampler'});
extension.setup();node=new Node();node.onNodeCreated();
assert.equal(node.selfliftTilingPanel.readOnly,true);
assert.equal(node.widgets.at(-1).options.serialize,false);
assert.ok(node.selfliftSeamCanvas);
assert.match(node.selfliftTilingPanel.value,/Not run/);
node.widgets[1].value='manual';node.widgets[1].callback();
assert.match(node.selfliftTilingPanel.value,/manual/);
assert.match(node.selfliftTilingPanel.value,/Settings changed/);
events.get('selflift-avatar-tiling')({detail:{node_id:'78',text:'RUNNING | axis=W | Effective: 4'}});
assert.match(node.selfliftTilingPanel.value,/RUNNING.*Effective: 4/);
node.onExecuted({selflift_tiling:['DONE | Effective: 4']});
assert.match(node.selfliftTilingPanel.value,/DONE/);
node.widgets[3].value='height';node.widgets[3].callback();
assert.match(node.selfliftTilingPanel.value,/axis=height/);
assert.doesNotMatch(node.selfliftTilingPanel.value,/DONE/);
node.onConfigure();
assert.match(node.selfliftTilingPanel.value,/Settings changed/);
events.get('selflift-avatar-tiling')({detail:{node_id:'79',text:'WRONG NODE'}});
assert.doesNotMatch(node.selfliftTilingPanel.value,/WRONG NODE/);
events.get('selflift-avatar-tiling')({detail:{node_id:'78',text:'FAILED'}});
assert.match(node.selfliftTilingPanel.value,/FAILED/);
console.log('Frontend callback checks passed: read-only/nonserialized panel, controls, live event, completion, stale reset, configure, node routing, failure.');

// Automatic planning must show the real count above the configuration summary.
node.widgets[1].value='auto';node.widgets[1].callback();
assert.match(node.selfliftTilingPanel.value.split('\n')[0], /pending/);
events.get('selflift-avatar-tiling')({detail:{node_id:'78',text:'RUNNING',plan:{tiles:3,mode:'auto',axis:4}}});
assert.match(node.selfliftTilingPanel.value.split('\n')[0], /Actual tiles: 3/);
assert.equal(node.widgets[1].value,'auto');
assert.equal(node.widgets[2].value,2); // Never overwrite the manual dropdown.
node.onExecuted({selflift_tiling:['DONE'],selflift_tiling_plan:[{tiles:3,mode:'auto',axis:4}]});
assert.match(node.selfliftTilingPanel.value.split('\n')[0], /Actual tiles: 3/);
events.get('selflift-avatar-tiling')({detail:{node_id:'78',text:'Preparing',plan:null}});
assert.match(node.selfliftTilingPanel.value.split('\n')[0], /pending/);
// Accommodate graphs keyed by numeric IDs.
globalThis.testApp.graph.getNodeById=(id)=>id===78?node:null;
events.get('selflift-avatar-tiling')({detail:{node_id:'78',text:'RUNNING',plan:{tiles:1}}});
assert.match(node.selfliftTilingPanel.value.split('\n')[0], /Actual tiles: 1.*no split/);
node.onExecuted({selflift_tiling:['DONE\nRequested / 请求: auto | Effective / 实际: 4']});
assert.match(node.selfliftTilingPanel.value.split('\n')[0], /Actual tiles: 4/);
node.widgets[3].callback();
assert.match(node.selfliftTilingPanel.value.split('\n')[0], /pending/);
node.widgets[0].value=false;node.widgets[0].callback();
assert.match(node.selfliftTilingPanel.value.split('\n')[0], /OFF/);

// Global-context controls and status.
const widget=(name)=>node.widgets.find((w)=>w.name===name);
node.widgets[0].value=true; node.widgets[4].value='global_stream'; node.widgets[4].callback();
assert.match(node.selfliftTilingPanel.value, /global_stream/);
assert.match(node.selfliftTilingPanel.value, /token_chunk=4096/);
assert.equal(widget('tiling_mode').type, 'selflift-hidden');
assert.notEqual(widget('highres_token_chunk').type, 'selflift-hidden');
events.get('selflift-avatar-tiling')({detail:{node_id:'78',text:'RUNNING',plan:{kind:'global',mode:'global_stream'}}});
assert.match(node.selfliftTilingPanel.value.split('\n')[0], /Global context/);
node.widgets[4].value='global'; node.widgets[4].callback();
assert.equal(widget('highres_weight_cache_gb').type, 'selflift-hidden');
node.widgets[4].value='spatial'; node.widgets[4].callback();
assert.notEqual(widget('tiling_mode').type, 'selflift-hidden');
assert.equal(widget('highres_token_chunk').type, 'selflift-hidden');
node.widgets[0].value=false; node.widgets[0].callback();
assert.equal(widget('highres_tiling_mode').type, 'selflift-hidden');
console.log('Automatic actual-count tests passed (live, complete, reset, cached text, numeric node ID, single tile).');
console.log('Global-context frontend checks passed (mode, status, control visibility).');
// Experimental spatial-anchor node gets the same plan panel and its extra controls.
class SpatialAnchorNode extends Node {
    constructor(){
        super();
        this.widgets.push({name:'spatial_anchor',value:'lowres_global'},
                          {name:'tiling_overlap',value:6});
    }
}
await extension.beforeRegisterNodeDef(SpatialAnchorNode,{name:'SelfLiftAvatarH3SpatialAnchoredSampler'});
const spatialAnchorNode = new SpatialAnchorNode();
spatialAnchorNode.onNodeCreated();
assert.match(spatialAnchorNode.selfliftTilingPanel.value,/anchor=lowres_global/);
assert.match(spatialAnchorNode.selfliftTilingPanel.value,/overlap=6/);
assert.match(spatialAnchorNode.selfliftTilingPanel.value,/spatial/);
assert.ok(spatialAnchorNode.selfliftSeamCanvas);
spatialAnchorNode.onExecuted({
    selflift_tiling:['DONE | seam diag'],
    selflift_seam_diag:[{
        axis:'W', length:24,
        regions:[[0,12],[10,24]],
        seams:[{boundary:1,cut:11,overlap:2,relative_diff:0.12}],
    }],
});
assert.equal(spatialAnchorNode.selfliftSeamData.seams.length,1);
assert.equal(spatialAnchorNode.selfliftSeamData.seams[0].relative_diff,0.12);
console.log('Spatial-anchor frontend checks passed (separate node, plan panel, anchor/overlap summary, seam diagram).');

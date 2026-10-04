import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
let extension;
globalThis.testModelOrderApp = {registerExtension(e) { extension = e; }, graph: {_nodes: []}};
const source = (await fs.readFile(new URL('../web/model_input_order.js', import.meta.url), 'utf8'))
    .replace('import { app } from "../../scripts/app.js";', 'const app = globalThis.testModelOrderApp;');
await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const names = ['low_res_model', 'positive', 'negative', 'vae', 'latent_image', 'sampler', 'sigmas', 'high_res_model'];
for (const type of ['SelfLiftAvatarH3Sampler', 'SelfLiftAvatarImageSampler']) {
    class Node {
        constructor() { this.type = type; this.inputs = names.map(name => ({name, link: null})); }
        onNodeCreated() { this.originalCalled = true; return 42; }
    }
    extension.beforeRegisterNodeDef(Node, {name: type});
    const fresh = new Node();
    assert.equal(fresh.onNodeCreated(), 42);
    assert.equal(fresh.originalCalled, true);
    assert.deepEqual(fresh.inputs.map(i => i.name), [names[0], names[7], ...names.slice(1, 7)]);
    const restored = new Node();
    const graph = {links: {}, _nodes: [restored]};
    restored.graph = graph;
    for (const [slot, input] of restored.inputs.entries()) {
        input.link = slot;
        graph.links[slot] = {target_slot: slot, origin_id: 100 + slot, origin_slot: 0};
    }
    globalThis.testModelOrderApp.graph = graph;
    extension.afterConfigureGraph();
    for (const [slot, input] of restored.inputs.entries()) {
        assert.equal(graph.links[input.link].target_slot, slot);
        assert.equal(graph.links[input.link].origin_id, 100 + names.indexOf(input.name));
    }
    const ordered = [...restored.inputs];
    extension.afterConfigureGraph();
    assert.deepEqual(restored.inputs, ordered);
    assert.deepEqual(restored.inputs.map(i => i.name), fresh.inputs.map(i => i.name));
}
class Other {}
extension.beforeRegisterNodeDef(Other, {name: 'SelfLiftAvatarH3TST'});
assert.equal(Other.prototype.onNodeCreated, undefined);
console.log('Model input order checks passed: both samplers, new/restored nodes, preserved links (including ID 0), idempotence, unrelated nodes.');

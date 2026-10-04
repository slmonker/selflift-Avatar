import { app } from "../../scripts/app.js";

const SAMPLERS = new Set(["SelfLiftAvatarH3Sampler", "SelfLiftAvatarImageSampler"]);

function orderModelInputs(node) {
    const inputs = node.inputs;
    if (!inputs) return;
    const low = inputs.findIndex((input) => input.name === "low_res_model");
    const high = inputs.findIndex((input) => input.name === "high_res_model");
    if (low < 0 || high < 0 || high === low + 1) return;
    const [input] = inputs.splice(high, 1);
    inputs.splice(inputs.findIndex((input) => input.name === "low_res_model") + 1, 0, input);
    // Links address slots by index; keep saved connections on the same input.
    for (let slot = 0; slot < inputs.length; slot++) {
        const linkId = inputs[slot].link;
        if (linkId == null) continue;
        const link = node.graph?.links[linkId];
        if (link) link.target_slot = slot;
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "selflift-Avatar.ModelInputOrder",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!SAMPLERS.has(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            orderModelInputs(this);
            return result;
        };
    },
    afterConfigureGraph() {
        // Wait until graph deserialization has restored all nodes and links.
        for (const node of app.graph._nodes) {
            if (SAMPLERS.has(node.type)) orderModelInputs(node);
        }
    },
});

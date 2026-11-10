// Adds a "show_preview" toggle (default off) to "Video To Extend (Create)".
//
// The preview itself comes from ComfyUI's frontend (because the video widget
// has the video_upload option). This file only hides or shows it.
//
// Design notes, for robustness against ComfyUI changes:
// - The toggle is frontend-only. Its state lives in node.properties, so it is
//   saved with the workflow but never sent to the backend; flipping it does not
//   invalidate ComfyUI's execution cache.
// - Hiding uses long-standing, public-ish flags: widget.hidden for the canvas
//   preview widget and node.hideOutputImages for the Vue node renderer.
// - Every step is wrapped in try/catch. If something here stops working after
//   a ComfyUI update, the preview simply shows as it would without this file.

import { app } from "../../scripts/app.js";

const NODE_TYPE = "VTE_Create";
const PROP = "vte_show_preview";
const TOGGLE_NAME = "show_preview";
const LOG_PREFIX = "[comfyui_video_to_extend]";

function warn(...args) {
    console.warn(LOG_PREFIX, ...args);
}

function isShown(node) {
    return !!node.properties?.[PROP];
}

function isPreviewWidget(w) {
    if (!w || w.name === TOGGLE_NAME) return false;
    if (w.name === "video-preview") return true;
    // Fallback in case the widget gets renamed: any DOM widget holding a video.
    const el = w.element;
    return !!(el && (el.tagName === "VIDEO" || el.querySelector?.("video")));
}

function applyVisibility(node) {
    try {
        const show = isShown(node);
        node.hideOutputImages = !show; // Vue node renderer

        let changed = false;
        for (const w of node.widgets ?? []) {
            if (!isPreviewWidget(w)) continue;
            if (!!w.hidden !== !show) {
                w.hidden = !show;
                changed = true;
            }
            const el = w.element;
            if (el?.style) el.style.display = show ? "" : "none";
            if (!show) el?.querySelectorAll?.("video").forEach((v) => v.pause?.());
        }

        if (changed && node.computeSize && node.setSize && node.size) {
            const minH = node.computeSize()[1];
            node.setSize([node.size[0], show ? Math.max(node.size[1], minH) : minH]);
        }
        node.setDirtyCanvas?.(true, true);
    } catch (e) {
        warn("could not update preview visibility", e);
    }
}

// Re-run the video widget's own change handler so the frontend re-evaluates
// its preview (needed for the Vue renderer, and to load a preview that was
// never loaded while hidden).
function refreshPreview(node) {
    try {
        const video = node.widgets?.find((w) => w.name === "video");
        video?.callback?.(video.value);
    } catch (e) {
        warn("could not refresh preview", e);
    }
}

function setShown(node, show) {
    node.properties = node.properties ?? {};
    node.properties[PROP] = !!show;
    const toggle = node.widgets?.find((w) => w.name === TOGGLE_NAME);
    if (toggle && toggle.value !== !!show) toggle.value = !!show;
    applyVisibility(node);
    refreshPreview(node);
    applyVisibility(node);
}

function setup(node) {
    try {
        node.properties = node.properties ?? {};
        if (node.properties[PROP] === undefined) node.properties[PROP] = false;

        if (!node.widgets?.some((w) => w.name === TOGGLE_NAME)) {
            const toggle = node.addWidget(
                "toggle",
                TOGGLE_NAME,
                isShown(node),
                (v) => setShown(node, v),
                { on: "shown", off: "hidden", serialize: false }
            );
            if (toggle) toggle.serialize = false; // keep out of widgets_values
        }
        applyVisibility(node);
    } catch (e) {
        warn("could not add preview toggle", e);
    }
}

function chain(proto, name, fn) {
    const orig = proto[name];
    proto[name] = function (...args) {
        const r = orig?.apply(this, args);
        try {
            fn.apply(this, args);
        } catch (e) {
            warn(`${name} hook failed`, e);
        }
        return r;
    };
}

app.registerExtension({
    name: "comfyui_video_to_extend.PreviewToggle",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;

        chain(nodeType.prototype, "onNodeCreated", function () {
            setup(this);
        });

        // Restore the saved state when a workflow is loaded.
        chain(nodeType.prototype, "onConfigure", function () {
            setup(this);
            const toggle = this.widgets?.find((w) => w.name === TOGGLE_NAME);
            if (toggle) toggle.value = isShown(this);
            applyVisibility(this);
        });

        // The preview widget is created later, after the video loads, so check
        // whenever the node's widget list changes.
        chain(nodeType.prototype, "onDrawForeground", function () {
            const count = this.widgets?.length ?? 0;
            if (count !== this.__vteWidgetCount) {
                this.__vteWidgetCount = count;
                applyVisibility(this);
            }
        });
    },
});

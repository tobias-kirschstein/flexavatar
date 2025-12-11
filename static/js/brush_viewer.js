import init, * as bindings from '../wasm/brush_app.js';

console.log("brush imported");

const isWebGPUSupported = 'gpu' in navigator;
// Display a warning if WebGPU is not supported.
const warningElement = document.getElementById('webgpu-warning');
const canvasElement = document.getElementById('brush_canvas');

// Prevent Brush from stealing focus.
const originalFocus = HTMLElement.prototype.focus;
HTMLElement.prototype.focus = function () {
    if (!this.matches('[id="brush_canvas"]')) {
        originalFocus.apply(this, arguments);
    }
};

// window.addEventListener("TrunkApplicationStarted", (e) => {
//     // Load with an initial URL.
//     console.log("starting")
//     // var start_url = "/static/files/abraham_short.ply"
//     // var start_url = "/static/files/cap-logo-black_woman_rasta_hair.ply"
//     var start_url = "/static/files/cap-logo-black_woman_rasta_hair.zip"
//     // const viewer = new window.wasmBindings.EmbeddedApp("brush_canvas", `?url=${start_url}&zen=true&focal=0.2&min_radius=1.3&max_radius=2.0&radius=1.5&min_pitch=-15.0&max_pitch=15.0&min_yaw=-45.0&max_yaw=45.0`)
//     const viewer = new window.wasmBindings.EmbeddedApp("brush_canvas", `?url=${start_url}&zen=true`)
//     window.viewer = viewer;
//
//     window.viewer.load_url(start_url);
//     const settings = new window.wasmBindings.CameraSettings(
//         0.3,           // fov_y (f64)
//         0.0, 0.0, 1.0, // position x, y, z (f32)
//         3.14, 0.0, 3.14, // 3.14, // euler_x, euler_y, euler_z (f32, radians)
//         1.0,           // focus_distance (f32)
//         2.0,           // speed_scale (f32 | null)
//         0.66, 1.33,    // min_focus_distance, max_focus_distance
//         //-180, 180,    // min_pitch, max_pitch
//         //-45, 45,    // min_yaw, max_yaw
//     );
//     window.viewer.set_camera_settings(settings);
// });
//
// if (isWebGPUSupported) {
//     const wasm = await init({module_or_path: './static/wasm/brush_app_bg.wasm'});
//     window.wasmBindings = bindings;
//     dispatchEvent(new CustomEvent("TrunkApplicationStarted", {detail: {wasm}}));
//     console.log("brush_app_bg.wasm imported");
// } else {
//     warningElement.style.display = 'block';
//     canvasElement.style.display = 'none';
// }




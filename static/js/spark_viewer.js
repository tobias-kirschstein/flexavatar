// import * as THREE from "three";
import * as THREE from "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.178.0/three.module.js";
// import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.178.0/build/three.module.js';
import { SplatMesh } from "@sparkjsdev/spark";
// import { SplatMesh } from "https://sparkjs.dev/releases/spark/0.1.10/spark.module.js";
import JSZip from "https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm";
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
// import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.178.0/examples/jsm/controls/OrbitControls.js';

const sparkContainer = document.getElementById("spark-viewer");
const containerWidth = sparkContainer.offsetWidth;
const containerHeight = sparkContainer.offsetHeight;

const loadingBarContainer = document.getElementById("loading-bar-container");
const loadingBar = document.getElementById("loading-bar");

// const camera = new THREE.PerspectiveCamera(45, containerWidth / containerHeight, 0.1, 1000);
const camera = new THREE.PerspectiveCamera(20, containerWidth / containerHeight, 0.1, 1000);
// camera.position.set(0, 0, 0.4);
camera.position.set(0, 0, 0.95);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(containerWidth, containerHeight);
sparkContainer.appendChild(renderer.domElement)

const scene = new THREE.Scene();
scene.background = new THREE.Color( 0xffffff );
let frames = [];

class AvatarSplatLoader {

    shouldStop = false;

    async loadFramesProgressively(zipURL) {
        // init loading bar
        loadingBarContainer.style.opacity = "1";
        loadingBarContainer.style.display = "block";
        loadingBar.style.width = "0%";


        const response = await fetch(zipURL);
        const zipData = await response.arrayBuffer();
        const zip = await JSZip.loadAsync(zipData);

        // sorted .ply files
        const plyFiles = Object.keys(zip.files)
            .filter(f => f.toLowerCase().endsWith(".spz"))
            .sort();

        const total = plyFiles.length;
        let loaded = 0;

        for (const fileName of plyFiles) {
            if (!this.shouldStop) {
                // Extract Blob asynchronously
                const blob = await zip.files[fileName].async("blob");
                const url = URL.createObjectURL(blob);

                // Create the SplatMesh
                const mesh = new SplatMesh({url});
                mesh.visible = false;
                scene.add(mesh);

                frames.push(mesh);

                // If this is the very first frame → show it immediately
                if (frames.length === 1) {
                    mesh.visible = true;
                }

                // Update loading bar
                loaded++;
                const pct = (loaded / total) * 100;
                loadingBar.style.width = pct + "%";

                // Hide bar when complete
                if (loaded === total) {
                    setTimeout(() => {
                        loadingBarContainer.style.opacity = "0";
                        loadingBarContainer.style.transition = "opacity 0.5s ease";
                        setTimeout(() => {
                            loadingBarContainer.style.display = 'none';
                        }, 600);
                    }, 300);
                }
                // Continue loop — next file loads in background while animation renders
            }
        }
    }

    stop() {
        this.shouldStop = true;
    }
}

async function loadAvatar(avatarName) {
    let avatarInputImage = document.getElementById("avatar-input-image");
    avatarInputImage.src = "./static/images/in_the_wild_inputs/" + avatarName + ".png";

    if (previousLoader != null) {
        previousLoader.stop();

        if (previousLoaderPromise != null) {
            await previousLoaderPromise
        }
    }

    //--------------------------------------------------------------------
    // 1. Load ZIP of 3D Gaussian splats
    //--------------------------------------------------------------------
    const zipURL = "./static/avatar_splats/" + avatarName + "_spark.zip";
    console.log("Loading " + zipURL);

    scene.clear();
    frames.length = 0;  // Also clear loaded splats
    let loader = new AvatarSplatLoader();
    previousLoaderPromise = loader.loadFramesProgressively(zipURL);  // start async loading in background
    previousLoader = loader;
}

let previousLoader = null;
let previousLoaderPromise = null;
window.loadAvatar = loadAvatar;

//--------------------------------------------------------------------
// 2. Animation Loop – cycle through mesh frames
//--------------------------------------------------------------------

// Setup mouse controls to orbit the camera around
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, -0.05);
controls.minDistance = 0.5;
controls.maxDistance = 1.1;
controls.enablePan = false;
controls.update();

let frameIndex = 0;
const frameDuration = 1000 / 30; // ~30 FPS
let lastFrameTime = 0;

loadAvatar("chatty-art-black_girl-holzstift_white_bg");
renderer.setAnimationLoop((time) => {
    if (frames.length > 1 && (time - lastFrameTime) > frameDuration) {
        // hide old frame
        for (const frame of frames) {
            frame.visible = false;
        }
        // frames[frameIndex % frames.length].visible = false;

        // advance index
        frameIndex = (frameIndex + 1) % frames.length;

        // show new frame
        frames[frameIndex % frames.length].visible = true;

        lastFrameTime = time;
    }

    renderer.render(scene, camera);
    controls.update();
});

function resizeViewer(event) {
    const containerWidth = sparkContainer.offsetWidth;
    const containerHeight = sparkContainer.offsetHeight;
    renderer.setSize(containerWidth, containerHeight);
}

window.addEventListener('resize', resizeViewer, true);
window.addEventListener('load', resizeViewer, true);

const avatarNames = [
    // "cap-logo-black_woman_rasta_hair",
    // "cap-logo-smiling_black_guy",
    // "cap-logo-white_woman_brown_hair",
    "cap-logo-white_woman_brown_wavy_hair",
    // "cap-logo-white_woman_brown_wavy_hair_againnnn",
    // "cap-logo-white_woman_redbrown_hair",
    "chatty-art-african_woman-3D_marble_sculpture",
    "chatty-art-african_woman-bronze_statue",
    // "chatty-art-asian_girl-comic_white_bg",
    "chatty-art-asian_girl-watercolors",
    // "chatty-art-asian_man-modern_flatillustration_white_bg",
    "chatty-art-black_girl-holzstift_white_bg",
    // "chatty-art-black_guy-comic_white_bg",
    // "chatty-art-redhead_woman-impressionist_impasto",
    // "chatty-art-white_boy-holzstift_white_bg",
    "chatty-art-white_woman_with_glasses-curly_hair-white_bg",
    //// "chatty-art-white_woman_with_glasses_long_hair-white_bg",
    "chatty-art-woman_with_glasses-spachteltechnik_white_bg",
    "chatty-asian_dude_with_glasses",
    //// "chatty-asian_woman_with_glasses",
    "chatty-black_girl_with_glasses",
    "chatty-black_man_with_glasses",
    //// "chatty-north_african_dude_with_glasses",
    //// "chatty-older_man_with_glasses",
    //// "chatty-white_boy_with_glasses",
    "chatty-white_older_woman_with_glasses",
    //// "chatty-white_woman_with_glasses",
    "chatty_art_middle_eastern_man_with_glasses_long_hair_white_bg",
    // "che_guevara",
    "einstein",
    "fairy_tale_woman_3",
    "fairy_tale_woman_flux_2",
    "fantasy_woman_glasses_2",
    "ffhq_00251",
    "ffhq_00273",
    "ffhq_00304",
    "ffhq_00380",
    "ffhq_00418",
    "ffhq_00427",
    // "ffhq_01037",
    //// "ffhq_01089",
    "ffhq_01104",
    "ffhq_01117",
    "ffhq_01125",
    "ffhq_01228",
    "ffhq_01238",
    // "ffhq_01245",
    "ffhq_01275",
    "ffhq_01331",
    "ffhq_01336",
    // "ffhq_01344",
    "ffhq_01356",
    //// "ffhq_01362",
    "ffhq_01548",
    // "flexavatar_cap1",
    // "flexavatar_cap2",
    "hadrian",
    // "meta_fairy_tale_woman",
    // "niessner",
    // "obama",
    "simon",
    "stylized_woman_glasses",
    //// "woman_glasses",
    // "woman_glasses2",
    //// "woman_glasses3",
    "woman_glasses4",
]

let avatarSelection = document.getElementById("avatar-selection");
for (const avatarName of avatarNames) {
    // avatarSelection.innerHTML += "<button onclick=\"window.loadAvatar('" + avatarName + "')\">" + avatarName + "</button>";
    avatarSelection.innerHTML += "<a href=\"javascript: onclick(window.loadAvatar('" + avatarName + "'))\" class=\"avatar-img-button is-1 column\"><img src=\"./static/images/in_the_wild_inputs/" + avatarName + ".png\" /></a>";
    console.log("Added " + avatarName);
}
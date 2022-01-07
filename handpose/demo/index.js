/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
import '@tensorflow/tfjs-backend-webgl';
import io from 'socket.io-client';
import feathers from '@feathersjs/feathers';
import socketio from '@feathersjs/socketio-client';

let feathersClient;
const clientColor = `#${Math.floor(Math.random()*16777215).toString(16)}`;

let hands = {};

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}
tfjsWasm.setWasmPaths({
  'tfjs-backend-wasm.wasm': `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/tfjs-backend-wasm.wasm`,
  'tfjs-backend-wasm-simd.wasm': `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/tfjs-backend-wasm-simd.wasm`,
  'tfjs-backend-wasm-threaded-simd.wasm': `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/tfjs-backend-wasm-threaded-simd.wasm`,
  });
let videoWidth, videoHeight, rafID, ctx, canvas, ANCHOR_POINTS,
    scatterGLHasInitialized = false, scatterGL, fingerLookupIndices = {
      thumb: [0, 1, 2, 3, 4],
      indexFinger: [0, 5, 6, 7, 8],
      middleFinger: [0, 9, 10, 11, 12],
      ringFinger: [0, 13, 14, 15, 16],
      pinky: [0, 17, 18, 19, 20]
    };  // for rendering each finger as a polyline

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 500;
const mobile = isMobile();

const state = {
  backend: 'webgl'
};

const stats = new Stats();
stats.showPanel(0);
// document.body.appendChild(stats.dom);

function drawKeypoints(keypoints, color) {
  const fingers = Object.keys(fingerLookupIndices);
  for (let i = 0; i < fingers.length; i++) {
    const finger = fingers[i];
    const points = fingerLookupIndices[finger].map(idx => keypoints[idx]);
    drawPath(points, false, color);
  }
}

function drawPath(points, closePath, color) {
  ctx.beginPath();
  ctx.lineWidth = 16;
  ctx.strokeStyle = color;
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
  ctx.closePath();
}

let model;

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width: mobile ? undefined : VIDEO_WIDTH,
      height: mobile ? undefined : VIDEO_HEIGHT
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();
  return video;
}
async function main() {
  info.textContent = 'Initializing things... this might take a while!';
  info.style.display = 'block';

  await tf.setBackend(state.backend);
  if (!tf.env().getAsync('WASM_HAS_SIMD_SUPPORT') && state.backend == "wasm") {
    console.warn("The backend is set to WebAssembly and SIMD support is turned off.\nThis could bottleneck your performance greatly, thus to prevent this enable SIMD Support in chrome://flags");
  }
  model = await handpose.load();
  let video;

  try {
    video = await loadVideo();
    pageInfo.style.display = 'block';
    info.textContent = 'Send this URL to someone, invite them to join you and show a hand to the webcam.';
    info.style.display = 'block';
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = e.message;
    info.style.display = 'block';
    // throw e;
  }

  const socket = io('https://talktothehand.hyper.fail');
  feathersClient = feathers();

  feathersClient.configure(socketio(socket));
  feathersClient
    .service('messages')
    .on('created', (message) => {
      const {predictions, user} = message;
      if (predictions && predictions.length && user !== clientColor) {
        hands[user] = predictions;
      }
    });

  // setupDatGui();

  videoWidth = video ? video.videoWidth : 640;
  videoHeight = video ? video.videoHeight : 480;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  if (video) {
    video.width = videoWidth;
    video.height = videoHeight;
  }

  ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = clientColor;
  ctx.fillStyle = clientColor;

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  // These anchor points allow the hand pointcloud to resize according to its
  // position in the input.
  ANCHOR_POINTS = [
    [0, 0, 0], [0, -VIDEO_HEIGHT, 0], [-VIDEO_WIDTH, 0, 0],
    [-VIDEO_WIDTH, -VIDEO_HEIGHT, 0]
  ];

  landmarksRealTime(video);
}

const landmarksRealTime = async (video) => {
  let lastFrame;
  async function frameLandmarks() {
    stats.begin();
    // const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    // if (lastFrame) {
    //   // ctx.putImageData(
    //   //   lastFrame, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width,
    //   //   canvas.height);
    //
    // }
    const fillStyle = ctx.fillStyle;
    ctx.beginPath();
    ctx.fillStyle = '#ffffff99';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    // // const fillStyle = ctx.fillStyle;
    // // ctx.fillStyle = '#ffffffcc';
    // // ctx.fillRect(0, 0, canvas.width, canvas.height);
    // // ctx.fillStyle = fillStyle;

    if (video) {
      const predictions = await model.estimateHands(video);
      if (predictions.length > 0) {
        feathersClient.service('messages').create({user: clientColor, predictions});
        const result = predictions[0].landmarks;
        drawKeypoints(result, clientColor + 'da');
      }
    }
    let colors = Object.keys(hands);
    for (const color of colors) {
      let predictions = hands[color];
      if (predictions.length) {
        let result = predictions[0].landmarks;
        drawKeypoints(result, color + 'da');
      }
    }
    // lastFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    stats.end();
    rafID = requestAnimationFrame(frameLandmarks);
  };

  frameLandmarks();
};

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

main();

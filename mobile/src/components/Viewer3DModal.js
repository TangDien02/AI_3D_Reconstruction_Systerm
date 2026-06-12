import React from 'react';
import { View, Text, TouchableOpacity, Modal, ActivityIndicator, StyleSheet, Platform } from 'react-native';
import { WebView } from 'react-native-webview';
import { StatusBar } from 'expo-status-bar';
import { C } from '../theme';
import { LogoMark } from './LogoMark';

export const build3DViewerHTML = (modelUrl) => `<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;background:#0A0E1A;overflow:hidden;font-family:sans-serif}
canvas{display:block;width:100%!important;height:100%!important}
#info{position:absolute;bottom:16px;left:50%;transform:translateX(-50%);
  background:rgba(0,0,0,0.6);color:#8A9DC4;font-size:11px;
  padding:6px 14px;border-radius:20px;pointer-events:none;white-space:nowrap}
#controls{position:absolute;top:16px;right:16px;display:flex;flex-direction:column;gap:8px}
.btn{background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);
  color:#fff;padding:8px 12px;border-radius:8px;font-size:12px;cursor:pointer;
  backdrop-filter:blur(4px);outline:none;transition:all 0.2s}
.btn:active{background:rgba(255,255,255,0.2)}
.btn.active{background:#2D6BE4;border-color:#5B8FEF}
#loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  color:#5B8FEF;font-size:13px;text-align:center}
.spinner{width:36px;height:36px;border:2px solid rgba(91,143,239,0.2);
  border-top-color:#5B8FEF;border-radius:50%;animation:spin 0.9s linear infinite;margin:0 auto 12px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div id="loading"><div class="spinner"></div>Loading 3D model...</div>
<div id="controls">
  <button id="wireframeBtn" class="btn" onclick="toggleWireframe()">Wireframe: OFF</button>
  <button id="autoRotateBtn" class="btn active" onclick="toggleAutoRotate()">Auto-Rotate: ON</button>
</div>
<div id="info">Drag to rotate · Pinch to zoom</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
var scene, camera, renderer, controls, model, isWireframe=false;
(function(){
  var loader=document.createElement('script');
  loader.src='https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js';
  loader.onload=function(){
    var ctrls=document.createElement('script');
    ctrls.src='https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js';
    ctrls.onload=initScene;
    document.head.appendChild(ctrls);
  };
  document.head.appendChild(loader);
})();

function toggleWireframe(){
  isWireframe = !isWireframe;
  var btn = document.getElementById('wireframeBtn');
  btn.innerText = 'Wireframe: ' + (isWireframe ? 'ON' : 'OFF');
  btn.className = isWireframe ? 'btn active' : 'btn';
  if(model){
    model.traverse(function(child){
      if(child.isMesh && child.material){
        var mats=Array.isArray(child.material)?child.material:[child.material];
        mats.forEach(function(mat){ mat.wireframe = isWireframe; });
      }
    });
  }
}

function toggleAutoRotate(){
  controls.autoRotate = !controls.autoRotate;
  var btn = document.getElementById('autoRotateBtn');
  btn.innerText = 'Auto-Rotate: ' + (controls.autoRotate ? 'ON' : 'OFF');
  btn.className = controls.autoRotate ? 'btn active' : 'btn';
}

function initScene(){
  var loadEl=document.getElementById('loading');
  scene=new THREE.Scene();
  scene.background=new THREE.Color(0x0A0E1A);

  var w=window.innerWidth,h=window.innerHeight;
  camera=new THREE.PerspectiveCamera(45,w/h,0.01,100);
  camera.position.set(0,0,2.5);

  renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.setSize(w,h);
  renderer.outputEncoding=THREE.sRGBEncoding;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure=1.8;
  document.body.appendChild(renderer.domElement);

  controls=new THREE.OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;
  controls.dampingFactor=0.08;
  controls.autoRotate=true;
  controls.autoRotateSpeed=1.2;

  scene.add(new THREE.AmbientLight(0xffffff,1.2));
  var dirLight1=new THREE.DirectionalLight(0xffffff,1.8);
  dirLight1.position.set(2,3,2);
  scene.add(dirLight1);
  var dirLight2=new THREE.DirectionalLight(0xaabbff,0.8);
  dirLight2.position.set(-2,1,-2);
  scene.add(dirLight2);

  var loader=new THREE.GLTFLoader();
  loader.load('${modelUrl}', function(gltf){
    loadEl.style.display='none';
    model=gltf.scene;
    model.traverse(function(child){
      if(child.isMesh && child.material){
        var mats=Array.isArray(child.material)?child.material:[child.material];
        mats.forEach(function(mat){
          if(mat.roughness!==undefined) mat.roughness=Math.max(0.3,mat.roughness);
          mat.wireframe = isWireframe;
        });
      }
    });
    var box=new THREE.Box3().setFromObject(model);
    var center=box.getCenter(new THREE.Vector3());
    var size=box.getSize(new THREE.Vector3());
    var maxDim=Math.max(size.x,size.y,size.z);
    var scale=1.6/maxDim;
    model.scale.setScalar(scale);
    model.position.sub(center.multiplyScalar(scale));
    scene.add(model);
  }, undefined, function(err){
    loadEl.innerHTML='<div style="color:#EF4444">Failed to load model</div>';
  });

  window.addEventListener('resize',function(){
    camera.aspect=window.innerWidth/window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth,window.innerHeight);
  });

  (function animate(){
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene,camera);
  })();
}
</script>
</body>
</html>`;

export const Viewer3DModal = ({ visible, modelUrl, onClose }) => (
  <Modal visible={visible} animationType="slide" statusBarTranslucent>
    <View style={S.viewerContainer}>
      <View style={S.viewerHeader}>
        <View style={S.viewerHeaderLeft}>
          <LogoMark size={24} />
          <Text style={S.viewerTitle}>3D Preview</Text>
        </View>
        <TouchableOpacity style={S.viewerCloseBtn} onPress={onClose}>
          <Text style={S.viewerCloseTxt}>✕</Text>
        </TouchableOpacity>
      </View>
      {modelUrl ? (
        <WebView
          source={{ html: build3DViewerHTML(modelUrl) }}
          style={S.webview}
          originWhitelist={['*']}
          javaScriptEnabled={true}
          domStorageEnabled={true}
          allowFileAccess={true}
          mixedContentMode="always"
        />
      ) : (
        <View style={S.viewerLoading}>
          <ActivityIndicator color={C.accentLight} size="large" />
          <Text style={S.viewerLoadingTxt}>Loading model...</Text>
        </View>
      )}
      <View style={S.viewerFooter}>
        <Text style={S.viewerHint}>Drag to rotate · Pinch to zoom</Text>
      </View>
    </View>
    <StatusBar style="light" />
  </Modal>
);

const S = StyleSheet.create({
  viewerContainer: { flex: 1, backgroundColor: C.bg },
  viewerHeader: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: Platform.OS === 'ios' ? 56 : 16,
    paddingBottom: 12,
    borderBottomWidth: 0.5, borderBottomColor: C.border, backgroundColor: C.bgCard,
  },
  viewerHeaderLeft: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  viewerTitle: { color: C.textPrimary, fontSize: 16, fontWeight: '700' },
  viewerCloseBtn: { width: 36, height: 36, borderRadius: 10, backgroundColor: C.bgCardAlt, alignItems: 'center', justifyContent: 'center', borderWidth: 0.5, borderColor: C.border },
  viewerCloseTxt: { color: C.textSecondary, fontSize: 15 },
  webview: { flex: 1, backgroundColor: C.bg },
  viewerLoading: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12 },
  viewerLoadingTxt: { color: C.textMuted, fontSize: 14 },
  viewerFooter: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 20, paddingVertical: 14,
    paddingBottom: Platform.OS === 'ios' ? 34 : 14,
    borderTopWidth: 0.5, borderTopColor: C.border, backgroundColor: C.bgCard,
  },
  viewerHint: { color: C.textMuted, fontSize: 12 },
});

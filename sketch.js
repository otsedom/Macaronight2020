// Collection of basic and more ellaborated CV demos to show interested young students

// Facemesh from mediapipe source https://awesomeopensource.com/project/LingDong-/handpose-facemesh-demos
// - use facemesh to track face skeleton
// - send to server via socket.io
// - update display with other users' faces from server

// First of all, shut glitch up about p5's global namespace pollution using this magic comment:
/* global describe p5 setup draw P2D WEBGL ARROW CROSS face MOVE TEXT WAIT HALF_PI PI QUARTER_PI TAU TWO_PI DEGREES RADIANS DEG_TO_RAD RAD_TO_DEG CORNER CORNERS RADIUS RIGHT LEFT CENTER TOP BOTTOM BASELINE POINTS LINES LINE_STRIP LINE_LOOP TRIANGLES TRIANGLE_FAN TRIANGLE_STRIP QUADS QUAD_STRIP TESS CLOSE OPEN CHORD PIE PROJECT SQUARE ROUND BEVEL MITER RGB HSB HSL AUTO ALT BACKSPACE CONTROL DELETE DOWN_ARROW ENTER ESCAPE LEFT_ARROW OPTION RETURN RIGHT_ARROW SHIFT TAB UP_ARROW BLEND REMOVE ADD DARKEST LIGHTEST DIFFERENCE SUBTRACT EXCLUSION MULTIPLY SCREEN REPLACE OVERLAY HARD_LIGHT SOFT_LIGHT DODGE BURN THRESHOLD GRAY OPAQUE INVERT POSTERIZE DILATE ERODE BLUR NORMAL ITALIC BOLD BOLDITALIC LINEAR QUADRATIC BEZIER CURVE STROKE FILL TEXTURE IMMEDIATE IMAGE NEAREST REPEAT CLAMP MIRROR LANDSCAPE PORTRAIT GRID AXES frameCount deltaTime focused cursor frameRate getFrameRate setFrameRate noCursor displayWidth displayHeight windowWidth windowHeight width height fullscreen pixelDensity displayDensity getURL getURLPath getURLParams pushStyle popStyle popMatrix pushMatrix registerPromisePreload camera perspective ortho frustum createCamera setCamera setAttributes createCanvas resizeCanvas noCanvas createGraphics blendMode noLoop loop push pop redraw applyMatrix resetMatrix rotate rotateX rotateY rotateZ scale shearX shearY translate arc ellipse circle line point quad rect square triangle ellipseMode noSmooth rectMode smooth strokeCap strokeJoin strokeWeight bezier bezierDetail bezierPoint bezierTangent curve curveDetail curveTightness curvePoint curveTangent beginContour beginShape bezierVertex curveVertex endContour endShape quadraticVertex vertex alpha blue brightness color green hue lerpColor lightness red saturation background clear colorMode fill noFill noStroke stroke erase noErase createStringDict createNumberDict storeItem getItem clearStorage removeItem select selectAll removeElements createDiv createP createSpan createImg createA createSlider createButton createCheckbox createSelect createRadio createColorPicker createInput createFileInput createVideo createAudio VIDEO AUDIO createCapture createElement deviceOrientation accelerationX accelerationY accelerationZ pAccelerationX pAccelerationY pAccelerationZ rotationX rotationY rotationZ pRotationX pRotationY pRotationZ pRotateDirectionX pRotateDirectionY pRotateDirectionZ turnAxis setMoveThreshold setShakeThreshold isKeyPressed keyIsPressed key keyCode keyIsDown movedX movedY mouseX mouseY pmouseX pmouseY winMouseX winMouseY pwinMouseX pwinMouseY mouseButton mouseIsPressed requestPointerLock exitPointerLock touches createImage saveCanvas saveGif saveFrames loadImage image tint noTint imageMode pixels blend copy filter get loadPixels set updatePixels loadJSON loadStrings loadTable loadXML loadBytes httpGet httpPost httpDo createWriter save saveJSON saveJSONObject saveJSONArray saveStrings saveTable writeFile downloadFile abs ceil constrain dist exp floor lerp log mag map max min norm pow round sq sqrt fract createVector noise noiseDetail noiseSeed randomSeed random randomGaussian acos asin atan atan2 cos sin tan degrees radians angleMode textAlign textLeading textSize textStyle textWidth textAscent textDescent loadFont text textFont append arrayCopy concat reverse shorten shuffle sort splice subset float int str boolean byte char unchar hex unhex join match matchAll nf nfc nfp nfs split splitTokens trim day hour minute millis month second year plane box sphere cylinder cone ellipsoid torus orbitControl debugMode noDebugMode ambientLight specularColor directionalLight pointLight lights lightFalloff spotLight noLights loadModel model loadShader createShader shader resetShader normalMaterial texture textureMode textureWrap ambientMaterial emissiveMaterial specularMaterial shininess remove canvas drawingContext*/
// Also socket.io, tensorflow and facemesh's:
/* global describe facemesh tf io*/
// Also in landmarks.js
/* global describe VTX7 VTX33 VTX68 VTX468 TRI7 TRI33 TRI68 TRI468*/
// now any other lint errors will be your own problem



// Mediapipe facemask setup
// A choice for number of keypoints: 7,33,68,468
// === bare minimum 7 points ===
// var VTX = VTX7;
// === important facial feature 33 points ===
// var VTX = VTX33;
// === standard facial landmark 68 points ===
 var VTX = VTX68;
// === full facemesh 468 points ===
//var VTX = VTX468;

// select the right triangulation based on vertices
var TRI = VTX == VTX7 ? TRI7 : (VTX == VTX33 ? TRI33 : (VTX == VTX68 ? TRI68 : TRI468))

var MAX_FACES = 4; //default 10

var facemeshModel = null; // this will be loaded with the facemesh model
                          // WARNING: do NOT call it 'model', because p5 already has something called 'model'

var videoDataLoaded = false; // is webcam capture ready?

var statusText = "Loading facemesh model...";

var myFaces = []; // faces detected in this browser
                  // currently facemesh only supports single face, so this will be either empty or singleton

//Masks
var glasses,potatoeyes,potatomouth,hair,rabbit;

//Images
var rgb, hola, holaRGB, holazoom,holaRGBzoom, holadif, holainv, holaedges;
var portada, temas;

//Icons
var iUpDown,iLeftRight,iMouseLeft,iMouseWheel;

// Images
//var capture; // webcam capture, managed by p5.js
let cam;
//var previousPixels;

// we need one extra createGraphics layer for the previous video frame
let pastFrame;

// shader variables
let sh_web,sh_rgb,sh_rgbinv,sh_gray,sh_edges, sh_edgesbw, sh_threshold,sh_diff,sh_alma,sh_almav2,sh_kernel;
var sh_enabled;

// this variable will hold our createGraphics layer
let shaderGraphics;

var rgbscale;

var mode, modemax, funmask, maskmode,maskmodemax;
let maskFrame;

var h, w;

var demolaunched; 

var startTime;

// Load the MediaPipe facemesh model assets.
facemesh.load().then(function(_model){
  console.log("model initialized.")
  statusText = "Model loaded."
  facemeshModel = _model;
})


//Based on https://github.com/aferriss/p5jsShaderExamples
function preload(){
  // load the shader
  sh_web = loadShader('effect.vert', 'webcam.frag');
  sh_rgb = loadShader('effect.vert', 'rgb.frag');
  sh_rgbinv = loadShader('effect.vert', 'rgb_inv.frag');
  sh_gray = loadShader('effect.vert', 'gray.frag');
  sh_threshold = loadShader('effect.vert', 'threshold.frag');
  sh_diff = loadShader('effect.vert', 'diff.frag');
  sh_edgesbw = loadShader('effect.vert', 'edgesBW.frag');
  sh_edges = loadShader('effect.vert', 'edges.frag'); 
  sh_alma = loadShader('effect.vert', 'almahaser.frag');
  sh_almav2 = loadShader('effect.vert', 'almahaserv2.frag');  
  sh_kernel = loadShader('effect.vert', 'kernel.frag');
  
  //Images
  rgb = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FLCD_RGB.jpg?v=1604492078393'); //https://commons.wikimedia.org/wiki/File:LCD_RGB.jpg
  portada = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FPortada.png?v=1606210554924');
  temas = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FTemas.png?v=1606216750831');
  holaRGB = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FHolaRGB.png?v=1606210587295');
  holaRGBzoom = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FHolaRGBzoom.png?v=1606218599329');
  hola = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FHola.png?v=1606210578177');
  holazoom = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FHolazoom.png?v=1606210605157');
  holadif = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FHolazoom_fotogramas.png?v=1606210610328');
  holainv = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FHolazoom_inverted.png?v=1606210617756');
  holaedges = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FHolazoomedges2.png?v=1606308080322');
    
  //Icons
  // UpDown https://ux.stackexchange.com/questions/108221/whats-the-difference-between-dropdown-caret-icon-and-unfold-more-icon
  // Mouse scroll https://freesvg.org/mouse-scroll
  // Mouse left click https://freesvg.org/mouse-leftclick
  iUpDown = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FUpDown.png?v=1605268506185');
  iLeftRight = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FLeftRight.png?v=1605268515970');
  iMouseLeft = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FMouseLeft.png?v=1605268526930');
  iMouseWheel = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FMouseWheel.png?v=1605268535736');
  
  //Masks full asset path
  glasses = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2Fgafas.png?v=1603278241408');
  potatoeyes = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FPotato_ojos.png?v=1604057501669');
  potatomouth = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2FPotato_boca.png?v=1604057511079');
  hair = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2Fhair.png?v=1604312423222');
  rabbit = loadImage('https://cdn.glitch.com/8f5e1723-ab6d-4ffd-9835-f35ff5be72f6%2Fconejo.png?v=1604312412668');
}

function setup() {
  startTime = millis();
  
  pixelDensity(1);
  
  //Main canvas
  createCanvas(windowWidth, windowHeight);  
  
  //Camera
  cam = createCapture(VIDEO);
  cam.size(640, 480);
  cam.hide();
  
  // Shaders require WEBGL mode to work
  shaderGraphics = createGraphics(cam.width, cam.height, WEBGL);
  shaderGraphics.noStroke();
  
  // this is to make sure the capture is loaded before asking facemesh to take a look
  // otherwise facemesh will be very unhappy
  cam.elt.onloadeddata = function(){
    console.log("video initialized");
    videoDataLoaded = true;
  }
  
  // the pastFrame layer doesn't need to be WEBGL
  pastFrame = createGraphics(cam.width, cam.height);
  
  // the maskFrame layer doesn't need to be WEBGL
  maskFrame = createGraphics(cam.width, cam.height);
  
  //Active demo selector
  mode = 0;
  modemax = 10;
  funmask = 0;
  maskmode = 0;
  maskmodemax = 6;
    
  //For rgm zoom
  rgbscale = 1.0;
  
  //Text features
  fill('blue');
  textAlign(CENTER);
  textSize(36);
  
  demolaunched = 0;
}

function draw() {
  background(255);
  
  var timeElapsed = (millis() - startTime);
  
  //if (timeElapsed<20000 && demolaunched == 0){
  if (demolaunched < 2){
    //if (timeElapsed>5000){ 
    if (demolaunched > 0){
      var wrate = width/temas.width;
      var hrate = height/temas.height;
      var scrate = wrate;
      if (hrate<wrate) scrate = hrate;
      
      image(temas, 0,0,temas.width*scrate,temas.height*scrate);
      
      
    }
    else{
      var wrate = width/temas.width;
      var hrate = height/temas.height;
      var scrate = wrate;
      if (hrate<wrate) scrate = hrate;
      
      image(portada, 0,0,portada.width*scrate,portada.height*scrate);
      
      if (timeElapsed>1000){
        //Forcing facemesh model load during launch
        if (facemeshModel && videoDataLoaded){ // model and video both loaded, 
          facemeshModel.pipeline.maxFaces = MAX_FACES;
          facemeshModel.estimateFaces(cam.elt).then(function(_faces){
            // we're faceling an async promise
            // best to avoid drawing something here! it might produce weird results due to racing

            myFaces = _faces.map(x=>packFace(x,VTX)); // update the global myFaces object with the detected faces

            // console.log(myFaces);
            if (!myFaces.length){
              // haven't found any faces
              statusText = "Show some faces!"
            }else{
              // display the confidence, to 3 decimal places
              statusText = "Confidence: "+ (Math.round(_faces[0].faceInViewConfidence*1000)/1000);
            }
          })
        }
      }
    }
  }
  else{
    //Acts according to current selected mode
    ModeProcessor(); 
  }
}

function windowResized(){
  resizeCanvas(windowWidth, windowHeight);
}

function mouseClicked() {  
  // If any
  if (sh_enabled){
    shaderGraphics.resetShader();
    sh_enabled = false;    
  }  
  
  //No mask
  funmask = 0;
  maskmode = 0;
  
  //
  rgbscale = 1.0;
  
  if (mouseButton === LEFT) {  
    if (demolaunched < 2) { // Click to avoid into
      demolaunched += 1;
    }
    else{
      mode = mode + 1;  
      if (mode > modemax) {
        mode = 0;
      }
    }
  }
}

function mouseWheel(event) {
  //Change zoom scale
  if (event.delta<0)
    rgbscale += 0.2;
  else
    rgbscale -= 0.2;
  
  //limit restrictions
  if (rgbscale < 1.0) rgbscale = 1.0;
  if (rgbscale > 20.0) rgbscale = 20.0;
  
  //uncomment to block page scrolling
  return false;
}

function keyPressed() {
    if (keyCode === LEFT_ARROW) {
      maskmode += 1;
      if (maskmode>maskmodemax)
        maskmode = 0;
    } else {
      if (keyCode === RIGHT_ARROW) {
        maskmode -= 1;
        if (maskmode<0)
          maskmode = maskmodemax;
      }
      else{
        if (keyCode === UP_ARROW || keyCode === DOWN_ARROW){
          if (sh_enabled){
            shaderGraphics.resetShader();
            sh_enabled = false;    
          }  
          //No mask
          funmask = 0;
          maskmode = 0;
          //resetting zoom  scale
          rgbscale = 1.0;
        }        
        
        if (keyCode === UP_ARROW) {  
          mode += 1;  
          if (mode > modemax) {
            mode = 0;          
          }          
        }
       else{
          if (keyCode === DOWN_ARROW) {  
          mode -= 1;  
          if (mode < 0) {
            mode = modemax;
          }          
        }
      }     
    }  
  }
}

function ModeProcessor(){
  var cad = "";
  
  switch (mode){
    case 0://Input RGB image   
        sh_enabled = true;    
        
        // shader() sets the active shader with our shader
        shaderGraphics.shader(sh_rgb);

        // passing cam as a texture
        sh_rgb.setUniform('tex0', cam);
        
        // passing cam as a texture
        sh_rgb.setUniform('scale', rgbscale);
        // also send the mouseX value but convert it to a number between 0 and 1
        sh_rgb.setUniform('mouseX', mouseX/cam.width);
        sh_rgb.setUniform('mouseY', mouseY/cam.height);

        // rect gives us some geometry on the screen
        shaderGraphics.rect(0,0,cam.width,cam.height);
        
        //Shows image
        image(shaderGraphics, 0, 0, cam.width, cam.height);
        
        //Image shown on the right area  
        if (rgbscale > 1.0){
          image(holaRGB, cam.width*1.1, cam.height*0.5-rgb.height/3);
          image(holaRGBzoom, cam.width*1.038, cam.height*0.5-rgb.height/3+holaRGB.height, holaRGBzoom.width*0.75, holaRGBzoom.height*0.75);
        }
        else{
          image(rgb, cam.width*1.1, cam.height*0.5-rgb.height/4, rgb.width/1.5,rgb.height/1.5);
        }

        //Active mode to be shown 
        cad = 'Imagen RGB';
        
        //RGB values shown on cursor
        //pixel RGB value
        shaderGraphics.loadPixels();     
      
        if (mouseX>0 && mouseX<cam.width && mouseY>0 && mouseY<cam.height){
            push();
            textSize(20);
            var color = get(mouseX,mouseY);
            textAlign(LEFT);

            var red = [255, 0, 0];
            var green = [0, 255, 0];
            var blue = [0, 0, 255];
            var string = [
                [color[0] + ' ', red],
                [color[1] + ' ', green],
                [color[2] + ' ', blue]
            ];
            drawtext(mouseX,mouseY, string );
            pop();
        }
        break;
        
    case 1: //Shader convert to gray
        sh_enabled = true;    
        
        // shader() sets the active shader with our shader
        shaderGraphics.shader(sh_gray);

        // passing cam as a texture
        sh_gray.setUniform('tex0', cam);

        // rect gives us some geometry on the screen
        shaderGraphics.rect(0,0,cam.width,cam.height);
        
        //Shows image
        image(shaderGraphics, 0, 0, cam.width, cam.height);
      
        //Image shown on the right area  
        image(hola, cam.width*1.1, cam.height*0.5-rgb.height/3);
        image(holazoom, cam.width*1.038, cam.height*0.5-rgb.height/3+holaRGB.height, holazoom.width*0.75, holazoom.height*0.75);
        
        
        //Active mode
        cad = 'Imagen grises';
      break;
        
    case 2: //Input RGB image   inverted
        sh_enabled = true;    
        
        // shader() sets the active shader with our shader
        shaderGraphics.shader(sh_rgbinv);

        // passing cam as a texture
        sh_rgbinv.setUniform('tex0', cam);

        // rect gives us some geometry on the screen
        shaderGraphics.rect(0,0,cam.width,cam.height);
        
        //SHows image
        image(shaderGraphics, 0, 0, cam.width, cam.height);
      
        //Image shown on the right area  
        image(holainv, cam.width*1.1, cam.height*0.5-rgb.height/3, holainv.width*0.6,holainv.width*0.6);
      

        //Active mode
        cad = 'Inversión';
        break;        
        
    case 3: //Shader edges
        sh_enabled = true;    
        
        // shader() sets the active shader with our shader
        shaderGraphics.shader(sh_edgesbw);

        // passing cam as a texture
        sh_edgesbw.setUniform('tex0', cam);
        
        // the size of one pixel on the screen
        sh_edgesbw.setUniform('stepSize', [1.0/width, 1.0/height]);

        // how far away to sample from the current pixel
        // 1 is 1 pixel away
        sh_edgesbw.setUniform('dist', 1.0);

        // rect gives us some geometry on the screen
        shaderGraphics.rect(0,0,cam.width,cam.height);
        
        //SHows image
        image(shaderGraphics, 0, 0, cam.width, cam.height);
      
        //Image shown on the right area  
        //image(holazoom, cam.width*1.038, cam.height*0.5-rgb.height/3+holaRGB.height, holazoom.width*0.75, holazoom.height*0.75);
        image(holaedges, cam.width*1.038, cam.height*0.5-rgb.height/3+holaRGB.height, holaedges.width*0.75, holaedges.height*0.75);
        
        //Active mode
        cad = 'Contornos 0/1';
      break;
        
    case 4: //Shader edges
        sh_enabled = true;    
        
        // shader() sets the active shader with our shader
        shaderGraphics.shader(sh_edges);

        // passing cam as a texture
        sh_edges.setUniform('tex0', cam);
        
        // the size of one pixel on the screen
        sh_edges.setUniform('stepSize', [1.0/width, 1.0/height]);

        // how far away to sample from the current pixel
        // 1 is 1 pixel away
        sh_edges.setUniform('dist', 1.0);

        /// rect gives us some geometry on the screen
        shaderGraphics.rect(0,0,cam.width,cam.height);
        
        //SHows image
        image(shaderGraphics, 0, 0, cam.width, cam.height);
      
      //Image shown on the right area  
        //image(holazoom, cam.width*1.038, cam.height*0.5-rgb.height/3+holaRGB.height, holazoom.width*0.75, holazoom.height*0.75);
        image(holaedges, cam.width*1.038, cam.height*0.5-rgb.height/3+holaRGB.height, holaedges.width*0.75, holaedges.height*0.75);
        
        //Active mode
        cad = 'Contornos';
      break;
        
    case 5: //Shader threshold
        sh_enabled = true;    
        
        // shader() sets the active shader with our shader
        shaderGraphics.shader(sh_threshold);

        // passing cam as a texture
        sh_threshold.setUniform('tex0', cam);
        
        // also send the mouseX value but convert it to a number between 0 and 1
        sh_threshold.setUniform('mouseX', mouseX/width);

        // rect gives us some geometry on the screen
        shaderGraphics.rect(0,0,cam.width,cam.height);
        
        //SHows image
        image(shaderGraphics, 0, 0, cam.width, cam.height);
      
      //Image shown on the right area  
        image(holazoom, cam.width*1.038, cam.height*0.5-rgb.height/3+holaRGB.height, holazoom.width*0.75, holazoom.height*0.75);
        
        //Active mode
        cad = 'Umbralizado';
      break;
        
    case 6: //Shader framediff
        cam.loadPixels();  // O falla en refrescar tras cambiar de modo
      
      
        sh_enabled = true;    
        
        // shader() sets the active shader with our shader
        shaderGraphics.shader(sh_diff);

        // passing cam as a texture
        sh_diff.setUniform('tex0', cam);
        
         // send the pastframe layer to the shader
        sh_diff.setUniform('tex1', pastFrame);
        
        // also send the mouseX value but convert it to a number between 0 and 1
        sh_diff.setUniform('mouseX', mouseX/width);

        /// rect gives us some geometry on the screen
        shaderGraphics.rect(0,0,cam.width,cam.height);
        
        //SHows image
        image(shaderGraphics, 0, 0, cam.width, cam.height);
      
      //Image shown on the right area  
        image(holadif, cam.width*1.1, cam.height*0.5-rgb.height/3, holadif.width*0.6,holadif.width*0.6);
        
        // draw the cam into the createGraphics layer at the very end of the draw loop
        // because this happens at the end, if we use it earlier in the loop it will still be referencing an older frame
        pastFrame.image(cam, 0,0, cam.width, cam.height);
        
        //Active mode
        cad = 'Diferencia de fotogramas';
      break;
        
      /*case 7: //Shader Alma Haser
        sh_enabled = true;    
        
        // shader() sets the active shader with our shader
        shaderGraphics.shader(sh_alma);

        // passing cam as a texture
        sh_alma.setUniform('tex0', cam);
        
        // also send the mouseX value but convert it to a number between 0 and 1
        sh_alma.setUniform('u_resolution', [width , height]);
        sh_alma.setUniform('mouseX', mouseX/width);
        sh_alma.setUniform('mouseY', mouseY/height);
        //sh_alma.set("u_resolution", float(width), float(height));
        //sh_alma.setUniform('mouseX', mouseX/width);
        //sh_alma.setUniform('mouseY', mouseY/height);

        // rect gives us some geometry on the screen
        shaderGraphics.rect(0,0,cam.width,cam.height);
        
        //SHows image
        image(shaderGraphics, 0, 0, cam.width, cam.height);
        
        //ACtive mode
        text('ALMA1', cam.width / 2, 36);
        cad = 'FACE';
      break;*/
        
      case 7: //Shader Alma Haser v2
        cam.loadPixels();  // O falla en refrescar tras cambiar de modo
      
        sh_enabled = true;    
        
        // shader() sets the active shader with our shader
        shaderGraphics.shader(sh_almav2);

        // passing cam as a texture
        sh_almav2.setUniform('tex0', cam);
        
        // also send the mouseX value but convert it to a number between 0 and 1
        sh_almav2.setUniform('u_resolution', [width , height]);
        sh_almav2.setUniform('mouseX', mouseX/width);
        
         // send the pastframe layer to the shader
        sh_almav2.setUniform('tex1', pastFrame);
        

        // rect gives us some geometry on the screen
        shaderGraphics.rect(0,0,cam.width,cam.height);
        
        //Shows image
        image(shaderGraphics, 0, 0, cam.width, cam.height);
        
        // draw the cam into the createGraphics layer at the very end of the draw loop
        // because this happens at the end, if we use it earlier in the loop it will still be referencing an older frame
        pastFrame.image(cam, 0,0, cam.width, cam.height);
        
        //Active mode
        cad = 'Inspirado en Alma Haser';
      break;
        
    /*case 8: //Shader webcam demo
        sh_enabled = true;    
        
        // shader() sets the active shader with our shader
        shaderGraphics.shader(sh_web);

        // passing cam as a texture
        sh_web.setUniform('tex0', cam);

        // rect gives us some geometry on the screen
        rect(posx,posy,cam.width, cam.height);
        
      break;*/
        
    case 8:  //Mediapipe facemesh        
      cam.loadPixels();  // O falla en refrescar tras cambiar de modo
  
      push();
      textSize(12);
      FaceMesh();   
      pop();      
      
      //Active mode
      cad = 'Detección de caras';
      break;
      
    case 9:  //Mediapipe facemesh        
      cam.loadPixels();  // O falla en refrescar tras cambiar de modo
  
      push();
      textSize(12);
      FaceMesh();   
      pop();      
      
      //Active mode
      cad = 'Clown';
      break;
        
    case 10:  //Mediapipe facemesh cwith additional fun effect
      funmask = 1; //active mask
      
      cam.loadPixels();  // O falla en refrescar tras cambiar de modo
  
      push();
      textSize(12);
      FaceMesh();   
      pop();  
            
      cad = 'Fun';
      break;
    }
  
  //Minimal UI comments and icons
  push();
  fill('black');
  textSize(20);
  textAlign(LEFT);
  text(cad, cam.width + 10, 20); 
  //text('Arriba/abajo o clic para cambiar',cam.width + 10, 40);
  text('modo',cam.width + 55, 51);
  image(iUpDown,cam.width +2+iMouseLeft.width*0.11,27,iUpDown.width*0.2,iUpDown.height*0.2);
  image(iMouseLeft,cam.width + 4 ,28,iMouseLeft.width*0.17,iMouseLeft.height*0.17);
  
  if (mode == 0){
     //text('Ruedita para zoom',cam.width + 10, 80)
    image(iMouseWheel,cam.width + 4 ,70,iMouseLeft.width*0.17,iMouseLeft.height*0.17);
    text('zoom',cam.width + 55, 95);
  }
  if (mode == 5 || mode == 6){
    text('Mueve el ratón',cam.width + 10, 90)
  }
  if (mode == 10){
    //text('Derecha/izquierda',cam.width + 10, 80)
    text('efecto',cam.width + 55, 95);
    image(iLeftRight,cam.width +6,70,iLeftRight.width*0.2,iLeftRight.height*0.2);
  }
  pop();
}


//Source https://stackoverflow.com/questions/52614829/p5-js-change-text-color-in-for-a-single-word-in-a-sentence
function drawtext( x, y, text_array ) {  
    var pos_x = x;
    for ( var i = 0; i < text_array.length; ++ i ) {
        var part = text_array[i];
        var t = part[0];
        var c = part[1];
        var w = textWidth( t );
        fill( c );
        text( t, pos_x, y);
        pos_x += w;
    }
}

function copyImage(src, dst) {
    var n = src.length;
    if (!dst || dst.length != n) dst = new src.constructor(n);
    while (n--) dst[n] = src[n];
    return dst;
}

// draw a face object returned by facemesh
function drawFaces(faces,filled){

  for (var i = 0; i < faces.length; i++){
    const keypoints = faces[i].scaledMesh;    
    
    //MASK if active
    if (funmask == 0){
      for (var j = 0; j < keypoints.length; j++) {
        const [x, y, z] = keypoints[j];
        push();
        if (mode == 9 ) stroke(128,0,0);
        //MASK POINTS
        circle(x,y,5);
        pop();
        //TEXT
        push();
        if (mode == 9 ) stroke(128,0,0);
        strokeWeight(1);
        text(j,x,y);
        pop()        
      }

      for (var j = 0; j < TRI.length; j+=3){
        var a = keypoints[TRI[j  ]];
        var b = keypoints[TRI[j+1]];
        var c = keypoints[TRI[j+2]];

        if (filled){
          var d = [(a[0]+b[0]+c[0])/6, (a[1]+b[1]+c[1])/6];
          var color = get(...d);
          fill(color);
          noStroke();
        }

        //Drawing mask triangles
        push();
        if (mode == 9 ) stroke(128,0,0);
        triangle(
          a[0],a[1],
          b[0],b[1],
          c[0],c[1],
        )
        pop();      
      }
      //Clown nose
      if (mode == 9){
        var le = keypoints[36]; // left eye corner
        var re = keypoints[45]; // right eye corner
        var esx = re[0]-le[0]; //Inter eye x to compute scale reduction
        //Nose location
        const [x, y, z] = keypoints[30];
        push();
        noStroke();
        fill(255,0,0);
        //MASK POINTS
        circle(x,y,esx/3);
        pop();
        
      }
        
    }
    else{ //Place fun mask on the propoer face location
      FaceMask(keypoints);
    }      
  }
}


// Fun effect aplied on the detected face
function FaceMask(keypoints){
  let camcopy = createImage(cam.width, cam.height);
  
  
    switch (maskmode){
        case 0: 
          //Rabbit
          DrawMaskonEyeBrows(keypoints, rabbit,0.25);  
        
          //Glasses
          DrawMaskonEyes(keypoints, glasses,0.7);       
         
          break;
        case 1: //Mr potato
          //Face mask border positions
          var faceidx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17];
          push();
          noStroke();
          fill(85, 79, 72);
          beginShape();
          for (var j = 0; j < faceidx.length; j++) {
            var pt = keypoints[faceidx[j]];
            vertex(pt[0], pt[1]);
          }          
          endShape(CLOSE);
          pop()
          DrawMaskonEyes(keypoints, potatoeyes,0.8);   
          DrawMaskonMouth(keypoints, potatomouth);   
        break;
        
        case 2: //Face mask
          //Face mask border positions
          var faceidx = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 30];
          push();
          noStroke();
          fill(200, 200, 200);
          beginShape();
          for (var j = 0; j < faceidx.length; j++) {
            var pt = keypoints[faceidx[j]];
            vertex(pt[0], pt[1]);
          }          
          endShape(CLOSE);
          pop()
        break;       
      case 3: //Based on CCV2019
        
        //Pink hair
        DrawMaskonEyes(keypoints, hair,0.3);   
        
        var pt1 = keypoints[16];
        var pt2 = keypoints[0];
        var facewidth = sqrt ( pow(pt1[0]-pt2[0],2) + pow(pt1[1]-pt2[1],2) );        
        
        //Earrings
        pt1 = keypoints[2];
        push();
        stroke(0,255,0);
        fill((0,255,0))
        ellipseMode(CENTER);
        strokeWeight(2);
        circle(pt1[0],pt1[1],facewidth/5);
        noFill();
        stroke(0,255,255);
        ellipse(pt1[0],pt1[1], 7 , 100 ) ;
        pop();
        
        pt1 = keypoints[14];
        push();
        stroke(0,255,0);
        fill((0,255,0))
        ellipseMode(CENTER);
        strokeWeight(2);
        circle(pt1[0],pt1[1],facewidth/5);
        noFill();
        stroke(0,255,255);
        ellipse(pt1[0],pt1[1], 7 , 100 ) ;
        pop();
        
        //Monocule
        pt1 = keypoints[41];
        push();
        stroke(153, 255, 255);
        //ellipseMode(CENTER);
        strokeWeight(4);
        circle(pt1[0],pt1[1],facewidth/4.5);
        pop();
        
        //Lips
        pt1 = keypoints[50];
        pt2 = keypoints[52];
        push();
        stroke(255,0,0);
        strokeWeight(10);
        arc(pt1[0],pt1[1],facewidth/10, facewidth/10,PI, 2*PI);
        arc(pt2[0],pt2[1],facewidth/10, facewidth/10,PI, 2*PI);        
        pop();
        pt1 = keypoints[66];
        
        push();
        fill(255,0,0);
        triangle(pt1[0]-8, pt1[1], pt1[0]+8, pt1[1],pt1[0], pt1[1]+8);
        pop();
        
        break;
        
      case 4: //Big face
        //Face mask border positions
        var faceidx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17];
        push();
        maskFrame.background(0);
        maskFrame.noStroke();
        maskFrame.fill(255);
        maskFrame.beginShape();
        var cx = 0, cy = 0;
        var xmin = cam.width,xmax = 0, ymin = cam.height,ymax = 0;
        for (var j = 0; j < faceidx.length; j++) {
          var pt = keypoints[faceidx[j]];
          maskFrame.vertex(pt[0], pt[1]);
          //Face center
          cx += pt[0];
          cy += pt[1];
          
          //face container
          if (pt[0]>xmax) {xmax=pt[0];}
          if (pt[1]>ymax) {ymax=pt[1];}
          if (pt[0]<xmin) {xmin=pt[0];}
          if (pt[1]<ymin) {ymin=pt[1];}
        }   
        cx = int(cx/faceidx.length);
        cy = int(cy/faceidx.length);
        maskFrame.endShape(CLOSE);
        pop();        
        
        //Intermediate image needed to operate
        camcopy.copy(cam, 0, 0, cam.width, cam.height, 0, 0, cam.width, cam.height);
        
       //Apply 3x3 filter
        cam.loadPixels();
        camcopy.loadPixels();
        maskFrame.loadPixels();
        for (var x = int(xmin); x < int(xmax); x++) {
          for (var y = int(ymin); y < int(ymax); y++) {
        //for (var x = int(xmin); x < cam.width; x++) {
          //for (var y = 0; y < cam.height; y++) {    
            var index = (x + y * cam.width)*4;
            //Only in mask area
            if (maskFrame.pixels[index]>0){                  
              //Big head
              //RGB of a face pixel
              let kx,ky;
              let valR = camcopy.pixels[index];
              let valG = camcopy.pixels[index+1];
              let valB = camcopy.pixels[index+2];
              
              //Face ccordinate duplicating according to its center
              var xpos = int(2*x - cx);
              var ypos = int(2*y - cy);
              
              //Inside the image 
              if (xpos>=0 && ypos>=0 && xpos+1<cam.width && ypos+1<cam.height){         
                //Replicates 2x2
                for (kx = 0; kx <= 1; kx++) {
                  let xpos2 = xpos + kx;
                  for (ky = 0; ky <= 1; ky++) {
                    let ypos2 = ypos + ky;
                    var indexK = (xpos2 + ypos2 * cam.width)*4;
                
                    //Copied in another location
                    cam.pixels[indexK] = valR;
                    cam.pixels[indexK+1] = valG;
                    cam.pixels[indexK+2] = valB;
                  }
                }               
              }
            }                 
          }
        }
        cam.updatePixels();
        
        image(cam, 0,0, cam.width, cam.height);
        break;
        
      case 5: //Creating image with the mask  
        //filter(BLUR,1);
        //Face mask border positions
        var faceidx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17];
        push();
        maskFrame.background(0);
        maskFrame.noStroke();
        maskFrame.fill(255);
        maskFrame.beginShape();
        var xmin = cam.width,xmax = 0, ymin = cam.height,ymax = 0;
        for (var j = 0; j < faceidx.length; j++) {
          var pt = keypoints[faceidx[j]];
          maskFrame.vertex(pt[0], pt[1]);
          
          //face container
          if (pt[0]>xmax) {xmax=pt[0];}
          if (pt[1]>ymax) {ymax=pt[1];}
          if (pt[0]<xmin) {xmin=pt[0];}
          if (pt[1]<ymin) {ymin=pt[1];}
        }   
        maskFrame.endShape(CLOSE);
        pop();
        //Shows the mask
        //image(maskFrame, cam.width,cam.height/2, cam.width, cam.height);
        
        //Intermediate image needed to operate
        camcopy.copy(cam, 0, 0, cam.width, cam.height, 0, 0, cam.width, cam.height);
        
        //let v = 1.0 / 9.0;
        // kernel is the 3x3 matrix of normalized values
        //let kernel = [[ v, v, v ], [ v, v, v ], [ v, v, v ]]; //structure for a generic 3x3 kernel
        //let kernel = [[ -1, 0, 1 ], [ -1, 0, 1 ], [ -1, 0, 1 ]],nv = 1.0,off = 1; //Edges
        let kernel = [[ -2, -1, 0 ], [ -1, 0, 1 ], [ 0, 1, 2 ]],nv = 1.0,off = 1; //Edgesv2
        //let kernel = [[ 1, 1, 1 ], [ 1, 1, 1 ], [ 1, 1, 1 ]],nv = 9.0,off = 1; //median
        //let kernel = [ [ 1, 1, 1, 1, 1, 1, 1 ], [ 1, 1, 1, 1, 1, 1, 1 ], [ 1, 1, 1, 1, 1, 1, 1 ], [ 1, 1, 1, 1, 1, 1, 1 ], [ 1, 1, 1, 1, 1, 1, 1 ], [ 1, 1, 1, 1, 1, 1, 1 ]],nv = 49.0,off = 3; //7x7 median
        //let kernel = [[ 21, 31, 21 ], [ 31, 48, 31 ], [ 21, 31, 21 ]],nv = 256.0,off = 1; //blur 3x3
        //let kernel = [[ 1, 4, 7, 4, 1], [ 4, 16, 26, 16, 4 ], [ 7, 26, 41, 26, 7], [ 4, 16, 26, 16, 4 ], [ 1, 4, 7, 4, 1]],nv = 273.0,off = 2; //blur 5x5
        
        //let kernel = [[ 1, 2, 1 ], [ 2, 4, 2 ], [ 1, 2, 1 ]], nv = 16.0, off = 1; //blur 3x3 
        
        //Apply 3x3 filter
        cam.loadPixels();
        camcopy.loadPixels();
        maskFrame.loadPixels();
        let ntot=0;
        for (var x = int(xmin); x < int(xmax); x++) {
          for (var y = int(ymin); y < int(ymax); y++) {
            var index = (x + y * cam.width)*4;
            //Only in mask area
            if (maskFrame.pixels[index+0]>0){    
              //Kernel filter 
              let sumR = 0, sumG = 0, sumB = 0, kx,ky; 
              if (x-off>=0 && y-off>=0 && x+off<cam.width && y+off<cam.height){  
                // kernel sum for the current pixel starts as 0 Adapted from source https://p5js.org/es/examples/image-blur.html
                // to consider all neighboring pixels we use a 3x3 array
                // and normalize these values
                // v is the normalized value

                // kx, ky variables for iterating over the kernel
                // kx, ky have three different values: -1, 0, 1
                for (kx = -off; kx <= off; kx++) {
                  let xpos = x + kx;
                  for (ky = -off; ky <= off; ky++) {
                    let ypos = y + ky;
                    //neighborhood
                    var indexK = int((xpos + ypos * cam.width)*4);

                    // get RGB values
                    let valR = camcopy.pixels[indexK];
                    let valG = camcopy.pixels[indexK+1];
                    let valB = camcopy.pixels[indexK+2];

                    // accumulate the  kernel sum
                    // kernel is a 3x3 matrix
                    // kx and ky have values -off,...,0,...,off
                    // if we add off to kx and ky, we get 0, 1, 2
                    // with that we can use it to iterate over kernel
                    // and calculate the accumulated sum
                    sumR += kernel[kx+off][ky+off] * valR;
                    sumG += kernel[kx+off][ky+off] * valG;
                    sumB += kernel[kx+off][ky+off] * valB;                    
                  }
                }
                // set the value of the edgeImg pixel to the kernel sum               
                cam.pixels[index] = (sumR/nv);
                cam.pixels[index+1] = (sumG/nv);
                cam.pixels[index+2] = (sumB/nv);   
                //cam.pixels[index+3] = 255;  
              }
            }                 
          }
        }
        cam.updatePixels();
        //maskFrame.updatePixels();
        
        image(cam, 0,0, cam.width, cam.height);
  
        break;
        
      case 6: //Based on Silvia and Gregor CCV2019
        
        //Mouth lines
        var pt1 = keypoints[48];
        var pt2 = keypoints[64];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(4);
        line(pt1[0],pt1[1],pt2[0],pt2[1]);
        pop();
        
        pt1 = keypoints[49];
        pt2 = keypoints[59];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(2);
        line(pt1[0],pt1[1],pt2[0],pt2[1]);
        pop();
        
        pt1 = keypoints[50];
        pt2 = keypoints[58];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(2);
        line(pt1[0],pt1[1],pt2[0],pt2[1]);
        pop();
        
        pt1 = keypoints[52];
        pt2 = keypoints[56];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(2);
        line(pt1[0],pt1[1],pt2[0],pt2[1]);
        pop();
        
        pt1 = keypoints[53];
        pt2 = keypoints[55];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(2);
        line(pt1[0],pt1[1],pt2[0],pt2[1]);
        pop();
        
        
        //Eyes
        pt1 = keypoints[37];
        pt2 = keypoints[40];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(2);
        line(pt1[0]-3,pt1[1]-3,pt2[0]+3,pt2[1]+3);
        pop();
        
        pt1 = keypoints[38];
        pt2 = keypoints[41];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(2);
        line(pt1[0]+3,pt1[1]-3,pt2[0]-3,pt2[1]+3);
        pop();
        
        pt1 = keypoints[44];
        pt2 = keypoints[47];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(2);
        line(pt1[0]+3,pt1[1]-3,pt2[0]-3,pt2[1]+3);
        pop();
        
        pt1 = keypoints[43];
        pt2 = keypoints[46];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(2);
        line(pt1[0]-3,pt1[1]-3,pt2[0]+3,pt2[1]+3);
        pop();
        
        //Nose
        pt1 = keypoints[28];
        pt2 = keypoints[31];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(3);
        line(pt1[0],pt1[1],pt2[0],pt2[1]);
        pop();
        
        pt1 = keypoints[31];
        pt2 = keypoints[35];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(3);
        line(pt1[0],pt1[1],pt2[0],pt2[1]);
        pop();
        
        pt1 = keypoints[35];
        pt2 = keypoints[28];
        push();
        stroke(random(255),random(255),random(255));
        strokeWeight(3);
        line(pt1[0],pt1[1],pt2[0],pt2[1]);
        pop();
        
        //Ellipse in mouth
        pt1 = keypoints[57];
        pt2 = keypoints[51];
        var pt3 = keypoints[54];
        var pt4 = keypoints[48];
        push();
        stroke(random(255),random(255),random(255));
        ellipseMode(CENTER);
        ellipse( (pt1[0]+pt2[0])/2 ,  (pt1[1]+pt2[1])/2, (pt3[0]-pt4[0]) , (pt1[1]-pt2[1]) ) ;
        pop();
        
        break;        
    }    
}

function DrawMaskonEyes(keypoints, img , scl){
  var ie = keypoints[27]; // inter eye
  var le = keypoints[36]; // left eye corner
  var re = keypoints[45]; // right eye corner
  var esx = re[0]-le[0]; //Inter eye x to compute scale reduction
  var factor=esx/(img.width*scl);
  push();
  //Face rotation angle
  let a = atan2(re[1]-le[1], re[0]-le[0]);
  // Translate to inter eye location
  translate(ie[0],ie[1]);
  //Rotate according to face
  rotate(a);
  //Draws mask
  var ofx = img.width/2*factor;
  var ofy = img.height/2*factor;
  image(img, -ofx,-ofy,img.width*factor,img.height*factor);
  pop();  
}

function DrawMaskonEyeBrows(keypoints, img , scl){
  var leb = keypoints[19]; // ledt eyebrow
  var reb = keypoints[24]; // ledt eyebrow
  var le = keypoints[36]; // left eye corner
  var re = keypoints[45]; // right eye corner
  var esx = re[0]-le[0]; //Inter eye x to compute scale reduction
  var factor=esx/(img.width*scl);
  push();
  //Face rotation angle
  let a = atan2(re[1]-le[1], re[0]-le[0]);
  // Translate to inter eyebrows location
  translate( (leb[0]+reb[0])/2 ,(leb[1]+reb[1])/2 );
  //Rotate according to face
  rotate(a);
  //Draws mask
  var ofx = img.width/2*factor;
  var ofy = img.height/2*factor;
  image(img, -ofx,-ofy,img.width*factor,img.height*factor);
  pop();  
}

function DrawMaskonMouth(keypoints, img){
  var im = keypoints[62]; // inyer eye
  var lm = keypoints[48]; // left eye corner
  var rm = keypoints[54]; // right eye corner
  var esx = rm[0]-lm[0]; //Inter eye x to compute scale reduction
  var factor=esx/(img.width*0.8);
  push();
  //Face rotation angle
  let a = atan2(rm[1]-lm[1], rm[0]-lm[0]);
  // Translate to inter eye location
  translate(im[0],im[1]);
  //Rotate according to face
  rotate(a);
  //Draws mask
  var ofx = img.width/2*factor;
  var ofy = img.height/2*factor;
  image(img, -ofx,-ofy,img.width*factor,img.height*factor);
  pop();
  
}


// reduces the number of keypoints to the desired set 
// (VTX7, VTX33, VTX68, etc.)
function packFace(face,set){
  var ret = {
    scaledMesh:[],
  }
  for (var i = 0; i < set.length; i++){
    var j = set[i];
    ret.scaledMesh[i] = [
      face.scaledMesh[j][0],
      face.scaledMesh[j][1],
      face.scaledMesh[j][2],
    ]
  }
  return ret;
}



function FaceMesh(){
  strokeJoin(ROUND); //otherwise super gnarly
      if (facemeshModel && videoDataLoaded){ // model and video both loaded, 
        facemeshModel.pipeline.maxFaces = MAX_FACES;
        facemeshModel.estimateFaces(cam.elt).then(function(_faces){
          // we're faceling an async promise
          // best to avoid drawing something here! it might produce weird results due to racing

          myFaces = _faces.map(x=>packFace(x,VTX)); // update the global myFaces object with the detected faces

          // console.log(myFaces);
          if (!myFaces.length){
            // haven't found any faces
            statusText = "Show some faces!"
          }else{
            // display the confidence, to 3 decimal places
            statusText = "Confidence: "+ (Math.round(_faces[0].faceInViewConfidence*1000)/1000);
          }
        })
      }

      // first draw the debug video and annotations
      push();
      //scale(0.5); // downscale the webcam capture before drawing, so it doesn't take up too much screen sapce
      image(cam, 0,0, cam.width, cam.height);
      noFill();
      stroke(255,0,0);
      drawFaces(myFaces); // draw my face skeleton
      pop();

      // now draw all the other users' faces (& drawings) from the server
      /*push()

      scale(2);
      strokeWeight(3);
      noFill();
      //drawFaces(myFaces);
      pop();

      push();
      fill(255,0,0);
      text(statusText,2,60);
      pop();*/
}
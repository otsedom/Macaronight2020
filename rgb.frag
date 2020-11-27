precision mediump float;

// grab texcoords from the vertex shader
varying vec2 vTexCoord;

// our texture coming from p5
uniform sampler2D tex0;
uniform float scale;
uniform float mouseX,mouseY;


void main() {

  vec2 uv = vTexCoord;
  // flip the y uvs
  uv.y = 1.0 - uv.y;
  
  //Scaling
  if (scale>1.0){
    vec2 pos = vec2(mouseX,mouseY);
    //vec2 pos = vec2(0.75,0.75);
    
    //Source https://www.shadertoy.com/view/lsyGDz
    vec2 zuv = uv/scale + pos;
    zuv -= pos/scale;

    // get the webcam as a vec4 using texture2D
    vec4 tex = texture2D(tex0, zuv);

    // output the grayscale value in all three rgb color channels
    gl_FragColor = tex;
  }
  else{
    // get the webcam as a vec4 using texture2D
    vec4 tex = texture2D(tex0, uv);

    // output the grayscale value in all three rgb color channels
    gl_FragColor = tex;
  }
}


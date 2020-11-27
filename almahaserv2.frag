// Inspired by Alma Haser https://scene360.com/art/112440/alma-hasers/

precision mediump float;

// grab texcoords from the vertex shader
varying vec2 vTexCoord;

// our texture coming from p5
uniform sampler2D tex0;
uniform sampler2D tex1;
uniform vec2 u_resolution;
uniform float mouseX;



//uniform vec2 texOffset; // sing processing
const vec2 texOffset = vec2(1.0, 1.0); // From https://www.nuomiphp.com/eplan/en/167810.html



float scale = 1.0;
float step = 0.025;

int flip = 0; // 1 uses flippes previous image, 0 does not

void main(void) {  
  vec2 stc = gl_FragCoord.xy/u_resolution;
  stc.y = 1.0 -stc.y;
  
  //step = mouseX/4.0; // Un comment to use a cell dimension mouse-affected
  
  vec2 uv = vTexCoord;
  
  // flip the y uvs
  uv.y = 1.0 - uv.y;
  vec2 uv2 = uv;
  uv2.x = 1.0 - uv2.x;
  
  // get the webcam as a vec4 using texture2D
  vec4 tex = vec4(0.0,0.0,0.0,1.0);
  //vec4 tex = texture2D(tex0, uv);
  
    // Alternating rows
	  if ( mod(stc.y*scale,2.0*step) <= step){
      //Columns
      // Positive displacement using step
      if ( mod(stc.x*scale,2.0*step) <= step)
        tex = texture2D(tex0, uv);
      else // Negative displacement
        if (flip == 0)
          tex = texture2D(tex1, uv);
        else
          tex = texture2D(tex1, uv2);
    }
    else
    {
      //Columns
      if ( mod(stc.x*scale,2.0*step) <= step)
        if (flip == 0)
          tex = texture2D(tex1, uv);
        else
          tex = texture2D(tex1, uv2);
      else // Negative displacement
        tex = texture2D(tex0, uv);

    }
    
  // output the grayscale value in all three rgb color channels
  gl_FragColor = tex;
  
}
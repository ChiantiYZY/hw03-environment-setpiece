#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;
uniform vec4 u_Size;

in vec4 fs_Pos;
out vec4 out_Col;


//random noise
float rand(float n){return fract(sin(n) * 43758.5453123);}
float noise(float p){
	float fl = floor(p);
  float fc = fract(p);
	return mix(rand(fl), rand(fl + 1.0), fc);
}

//perlin noise
vec2 fade(vec2 t) {return t*t*t*(t*(t*6.0-15.0)+10.0);}

vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}

float perlin_noise(vec2 P){
  vec4 Pi = floor(P.xyxy) + vec4(0.0, 0.0, 1.0, 1.0);
  vec4 Pf = fract(P.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);
  Pi = mod(Pi, 289.0); // To avoid truncation effects in permutation
  vec4 ix = Pi.xzxz;
  vec4 iy = Pi.yyww;
  vec4 fx = Pf.xzxz;
  vec4 fy = Pf.yyww;
  vec4 i = permute(permute(ix) + iy);
  vec4 gx = 2.0 * fract(i * 0.0243902439) - 1.0; // 1/41 = 0.024...
  vec4 gy = abs(gx) - 0.5;
  vec4 tx = floor(gx + 0.5);
  gx = gx - tx;
  vec2 g00 = vec2(gx.x,gy.x);
  vec2 g10 = vec2(gx.y,gy.y);
  vec2 g01 = vec2(gx.z,gy.z);
  vec2 g11 = vec2(gx.w,gy.w);
  vec4 norm = 1.79284291400159 - 0.85373472095314 * 
    vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11));
  g00 *= norm.x;
  g01 *= norm.y;
  g10 *= norm.z;
  g11 *= norm.w;
  float n00 = dot(g00, vec2(fx.x, fy.x));
  float n10 = dot(g10, vec2(fx.y, fy.y));
  float n01 = dot(g01, vec2(fx.z, fy.z));
  float n11 = dot(g11, vec2(fx.w, fy.w));
  vec2 fade_xy = fade(Pf.xy);
  vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
  float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
  return 2.3 * n_xy;
}


//Worley
vec3 random3( vec3 p ) {
    return fract(sin(vec3(dot(p,vec3(127.1, 311.7, 191.999)),
                          dot(p,vec3(269.5, 183.3, 765.54)),
                          dot(p, vec3(420.69, 631.2,109.21))))
                 *43758.5453);
}

float WorleyNoise3D(vec3 p)
{
    // Tile the space
    vec3 pointInt = floor(p);
    vec3 pointFract = fract(p);

    float minDist = 1.0; // Minimum distance initialized to max.

    // Search all neighboring cells and this cell for their point
    for(int z = -1; z <= 1; z++)
    {
        for(int y = -1; y <= 1; y++)
        {
            for(int x = -1; x <= 1; x++)
            {
                vec3 neighbor = vec3(float(x), float(y), float(z));

                // Random point inside current neighboring cell
                vec3 point = random3(pointInt + neighbor);

                // Animate the point
                //point = 0.5 + 0.5 * sin(u_Time * 0.01 + 6.2831 * point); // 0 to 1 range

                // Compute the distance b/t the point and the fragment
                // Store the min dist thus far
                vec3 diff = neighbor + point - pointFract;
                float dist = length(diff);
                minDist = min(minDist, dist);
            }
        }
    }
    return minDist;
}

float worleyFBM(vec3 uv) {
    float sum = 0.f;
    float freq = 4.f;
    float amp = 0.5;
    for(int i = 0; i < 5; i++) {
        sum += WorleyNoise3D(uv * freq) * amp;
        freq *= 2.f;
        amp *= 0.5;
    }
    return sum;
}


float impulse (float k, float x)
{
      float h = k * x;
      return h * exp(1.f - h);
}
mat4 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
        vec4(c, -s, 0, 0),
        vec4(s, c, 0, 0),
        vec4(0, 0, 1, 0),
        vec4(0, 0, 0, 1)
    );
}
mat4 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
        vec4(c, 0, -s, 0),
        vec4(0, 1, 0, 0),
        vec4(s, 0, c, 0),
        vec4(0, 0, 0, 1)
    );
}
mat4 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return mat4(
        vec4(1, 0, 0, 0),
        vec4(0, c, s, 0),
        vec4(0, -s, c, 0),
        vec4(0, 0, 0, 1)
    );
}
float trianlgeWave(float x, float freq, float amplitude)
{
    //x = m - abs(i % (2*m) - m)

    //return x - abs(float(int(freq) % int(2.f * x)) - x);

    //y = abs((x++ % 6) - 3);
    //return float(abs((float((int(x)+1) % 6)) - 3.f));
    return float(abs(int(x * freq) % int(amplitude) - int((0.5 * amplitude))));
}
float dot2( in vec2 v ) { return dot(v,v); }




float sdPlane( vec3 p, vec4 n )
{
  // n must be normalized
  return dot(p,n.xyz) + n.w;
}
vec3 opRep( vec3 p, vec3 c )
{
    return mod(p,c)-0.5*c;
}
float opSmoothSubtraction( float d1, float d2, float k ) 
{
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h); 

    return max(-d2, d1);
}
float opSmoothUnion( float d1, float d2, float k )
{
	float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
	return mix( d2, d1, h ) - k*h*(1.0-h);
}
float opSmoothIntersection( float d1, float d2, float k )
{
	float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
	return mix( d2, d1, h ) + k*h*(1.0-h);
}
vec3 opTwist( vec3 p )
{
    float  c = cos(10.0*p.y+10.0);
    float  s = sin(10.0*p.y+10.0);
    mat2   m0 = mat2(1.f, -1.f, 1.f, 1.f);
    mat2   m = mat2(c,-s,s,c);

    return vec3(m * p.xz,p.y);
}
vec3 opCheapBend( vec3 p )
{
    float step = smoothstep(-u_Size.w, u_Size.w, cos(u_Time * 0.05 + 3.f * p.x) * 0.01);
    float c = sin(step);
    float s = sin(step);
    mat2  m = mat2(c,-s,s,c);
    vec2 r = m * p.xy;
    vec3  q = vec3(r.xy, p.z);
    return q;
}

float onion( in float d, in float h )
{
    return abs(d)-h;
}



float sphere(vec3 p, float r)
{
    return length(p) - r;
}
float sdHexPrism( vec3 p, vec2 h )
{
    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
    p = abs(p);
    p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
    vec2 d = vec2(
       length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
       p.z-h.y );
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}
float sdTorus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}
float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0))
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf 
}
float sdCone( in vec3 p, in vec3 c )
{
    vec2 q = vec2( length(p.xz), p.y );
    float d1 = -q.y-c.z;
    float d2 = max( dot(q,c.xy), q.y);
    return length(max(vec2(d1,d2),0.0)) + min(max(d1,d2), 0.);
}
float sdCappedCone( in vec3 p, in float h, in float r1, in float r2 )
{
    vec2 q = vec2( length(p.xz), p.y );
    
    vec2 k1 = vec2(r2,h);
    vec2 k2 = vec2(r2-r1,2.0*h);
    vec2 ca = vec2(q.x-min(q.x,(q.y < 0.0)?r1:r2), abs(q.y)-h);
    vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot2(k2), 0.0, 1.0 );
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s*sqrt( min(dot2(ca),dot2(cb)) );
}
float sdCylinder( vec3 p, vec2 h )
{
    vec2 d = abs(vec2(length(p.xz),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}
float sdRoundBox( vec3 p, vec3 b, float r )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0)) - r
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf 
}
float sdRoundCone( vec3 p, float r1, float r2, float h )
{
    vec2 q = vec2( length(p.xz), p.y );
    
    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));
    
    if( k < 0.0 ) return length(q) - r1;
    if( k > a*h ) return length(q-vec2(0.0,h)) - r2;
        
    return dot(q, vec2(a,b) ) - r1;
}
float sdModPolar(inout vec2 p, float repetitions) {
    float angle = 2.*3.14/repetitions;
    float a = atan(p.y, p.x) + angle/2.;
    float r = length(p);
    float c = floor(a/angle);
    a = mod(a,angle) - angle/2.;
    p = vec2(cos(a), sin(a))*r;
    // For an odd number of repetitions, fix cell index of the cell in -x direction
    // (cell index would be e.g. -5 and 5 in the two halves of the cell):
    if (abs(c) >= (repetitions/2.)) c = abs(c);
    return c;
}


//engine
vec2 obj1(vec3 pos)
{

    mat4 r = rotateY(3.14  / 6.f); 

    
    pos -= vec3(0.f, 2.5f, 0.f);
    pos = vec3(r * vec4(pos, 1.0));
    pos /= 3.0;

    float d = 1e10;

    //top
    float d1 = sphere(pos - vec3(0.f, 1.8, 0.f), 1.7f);
    d1 = d1 - 0.2 * sin(10.f * pos.x)*sin(8.f * pos.y)*sin(6.f * pos.z);
    float d2 = sdCylinder(pos + vec3(0.f, 0.5f, 0.f), vec2(2.f, 2.5f));
    vec3 box_pos = opRep(vec3(atan(pos.x, pos.z), pos.y, 0.5 * length(pos)) , vec3(1.0, 0.0, 1.0));

   
    float d3 = sdBox(box_pos, vec3(0.2, 2.0, 0.1));

    d3 = min(d, opSmoothUnion(d3, d2, 0.1));
    d3 = min(d, opSmoothSubtraction(d1, d3, 0.1));

    float d4 = sdBox(pos, vec3(3.2f, 3.f, 3.f));
    d = min(d, opSmoothIntersection(d3, d4, 0.1));

    //base
    float d5 = sdCappedCone(pos + vec3(0.f, 4.0f, 0.f), 1.5, 8.0, 4.0);
    float d6 = sdCappedCone(pos + vec3(0.f, 3.6f, 0.f), 1.4, 7.0, 3.0);

    d6 = min(d, opSmoothSubtraction(d6, d5, 0.1));

    d = min(d, opSmoothUnion(d, d6, 0.1));


    //base_side
   // d = min(d, d3);
    return vec2(d * 3.0, 1.f);
}

//cone base
vec2 obj2(vec3 pos)
{
    pos /= 3.0;
    pos += vec3(0.0, 5.5, 0.0);
    pos *= 0.8;
    pos.x *= 0.8;

    float d = 1e10;
    mat4 r = rotateX(3.14 / 2.f);
    mat4 rr = rotateY(3.14 * 3.f / 6.f);
    mat4 rrr = rotateY(3.14 / 6.f);
    mat4 rrrr = rotateY(3.14 * 5.f / 6.f);
    //float d1 = sdCylinder(vec3(r * vec4(pos, 1.f)), vec2(1.f, 7.f));
    float d2 = sdCylinder(vec3(r * rr * vec4(pos, 1.f)), vec2(1.8f, 7.f));
    float d3 = sdCylinder(vec3(r * rrr * vec4(pos, 1.f)), vec2(1.8f, 7.f));
    float d4 = sdCylinder(vec3(r * rrrr * vec4(pos, 1.f)), vec2(1.8f, 7.f));
    //d = min(d, d1);
    d = min(d, d2);
    d = min(d, d3);
    d = min(d, d4);

    return vec2(d * 3.0 / 0.8, 1.f);
}

//floor
vec2 obj5(vec3 pos)
{

    float d = 1e10;

   // float worley_Offset = worleyFBM(vec3(pos.y) * 0.05);

    d = min(d, sdPlane(pos, normalize(vec4(0.0, 1.0, 0.0, 0.0))));

    return vec2(d, 3.f);

} 

//base
vec2 obj3(vec3 pos)
{
    pos /= 3.0;
    float d = 1e10;
    float d5 = sdCylinder(pos + vec3(0.0, 4.5, 0.0), vec2(3.8f, .3f));
    d = min(d, d5);

    return vec2(d * 3.0, 1.f);
}

//turnnel
vec2 obj4(vec3 pos)
{
    pos /= 3.0;
    pos += vec3(0.0, 5.5, 0.0);
    pos *= 0.8;

    float d = 1e10;
    mat4 r = rotateX(3.14 * 4.3f / 6.f);
    mat4 rr = rotateY(3.14 * 2.15f / 6.f);
    mat4 rrr = rotateY(-3.14 * 2.15f / 6.f);
    mat4 rrrr = rotateY(3.14 * 5.5f / 6.f);

    vec3 pos1 = vec3(r * rr * vec4(pos, 1.f));
    pos1 += vec3(0.0, -2.5, -4.5);
    float d2 = sdBox(pos1, vec3(0.5f, 2.f, 0.05));
    d = min(d, d2);

    vec3 pos2 = vec3(r * rrr * vec4(pos, 1.f));
    pos2 += vec3(0.0, -2.5, -4.5);
    float d3 = sdBox(pos2, vec3(0.5f, 2.f, 0.05));
    d = min(d, d3);


    return vec2(d * 3.0 / 0.8, 4.f);

}



//torus
vec2 obj6(vec3 pos)
{
    pos = pos + vec3(-7.5, -11.f, -2.0);

    pos = pos / 5.f;

    vec3 pos1 = vec3(rotateX(3.14159 / 2.f) * vec4(pos, 1.f));
    vec3 pos2 = opTwist(pos);
    float d = 1e10;


    //float d1 = sdTorus(pos1, vec2(1.f, 0.1));
    float d2 = sdTorus(pos2, vec2(0.3f, 0.1));

    //float d_mix = mix(d1, d2, abs(sin(u_Time * 0.01)));

    d = min(d, d2);

    return vec2(d * 5.f, 6.f);

}

//shelf
vec2 obj8(vec3 pos)
{
    float d = 1e10;
    float d1 = sdBox(pos + vec3(0.f, -5.f, 0.f), vec3(15.f, .1f, 4.f));
    float d2 = sdBox(pos + vec3(0.f, 10.f, 0.f), vec3(15.f, .1f, 4.f));
    float d3 = sdBox(pos + vec3(0.f, -20.f, 0.f), vec3(15.f, .1f, 4.f));
float d4 = sdBox(pos + vec3(0.f, -5.f, 0.f), vec3(.1f, 15.f, 4.f));
    float d5 = sdBox(pos + vec3(-15.f, -5.f, 0.f), vec3(.1f, 15.f, 4.f));
    float d6 = sdBox(pos + vec3(15.f, -5.f, 0.f), vec3(.1f, 15.f, 4.f));
    


    d = min(d, d1);
    d = min(d, d2);
    d = min(d, d3);
     d = min(d, opSmoothUnion(d, d4, 0.01));
    d = min(d, opSmoothUnion(d, d5, 0.01));
    d = min(d, opSmoothUnion(d, d6, 0.01));
    

    return vec2(d, 8.f);
}

vec2 obj7(vec3 pos)
{
    float d = 1e10;
    
    float d7 = sdBox(pos + vec3(0.f, -5.f, 4.f), vec3(15.f, 15.f, .1f));

   
    d = min(d, opSmoothUnion(d, d7, 0.01));

    return vec2(d, 7.f);
}

vec2 mapObj1(vec3 pos)
{
    float d = 1e10;
    float d_1 = sdBox(pos, vec3(15.f, 15.f, 15.f));

    d = min(d, d_1);
    return vec2(d, 8.f);
}

vec2 mapObj2(vec3 pos)
{
    float d = 1e10;
    float d1 = sdBox(pos, vec3(3.f));

    d = min(d, d1);
    return vec2(d, 8.f);
}

vec2 mapObj3(vec3 pos)
{
    float d = 1e10;
    float d1 = sdBox(pos + vec3(7.5, -12.5f, -2.f), vec3(5.f));

    d = min(d, d1);
    return vec2(d, 8.f);
}

vec2 mapObj4(vec3 pos)
{
    float d = 1e10;
    float d1 = sdBox(pos + vec3(-7.5, -12.5f, -2.f), vec3(7.f, 7.f, 5.f));

    d = min(d, d1);
    return vec2(d, 8.f);
}





vec2 map(vec3 pos)
{

    float d = 1e10;
   

    float tag = 1.f;

    // float box_d1 = mapObj1(pos).x;
    // float box_d2 = mapObj2(pos).x;
    // float box_d3 = mapObj3(pos).x;
    // float box_d4 = mapObj4(pos).x;


        //float height = perlin_noise((vec2(pos.x + pos.y, pos.z + pos.y) ) / 4.f);
        //float height_forest = perlin_noise((vec2(fs_Pos.x, fs_Pos.z) ) / 3.f);
       float worley_Offset = worleyFBM(pos * 0.05);
   
    //float offset = noise(pos.y * 10.0);

   float d1 = obj1(pos).x;
   float d2 = obj2(pos).x;
   float d3 = obj3(pos).x;
   float d4 = obj4(pos).x;
   float d5 = obj5(pos+ vec3(0.0, 15.0 + worley_Offset, 0.0)).x;

    d1 = min(d,  opSmoothSubtraction(d2, d1, 0.1));
    d1 = min(d, opSmoothUnion(d1, d3, 0.1));
    if(d1 < d)
    {
        d = d1;
        tag = obj1(pos).y;
    }

    d4 = min(d, opSmoothUnion(d, d4, 0.1));

    if(d4 < d)
    {
        d = d4;
        tag = obj4(pos).y;
    }



    if(d5 < d)
    {
        d = d5;
        tag = obj5(pos).y;
    }


    
  // d = min(d, d2);
    return vec2(d, tag);

}

vec3 normal(vec3 p) 
{
    float EPSILON = 0.1f;
    return normalize(vec3(
        map(vec3(p.x + EPSILON, p.y, p.z)).x - map(vec3(p.x - EPSILON, p.y, p.z)).x,
        map(vec3(p.x, p.y + EPSILON, p.z)).x - map(vec3(p.x, p.y - EPSILON, p.z)).x,
        map(vec3(p.x, p.y, p.z + EPSILON)).x - map(vec3(p.x, p.y, p.z - EPSILON)).x
    ));
}




vec2 dist(vec3 dir, vec3 eye)
{
  float depth = 0.001f;
  float tag;
for (int i = 0; i < 64; i++) {
    //float dist = sdHexPrism(eye + depth * dir, vec2(3.f, 3.f));
    //float dist = sdTorus(eye + depth * dir, vec2(3.f, 5.f));
    //float dist = sphere(eye + depth * dir);
    float dist = map(eye + depth * dir).x;
    tag = map(eye + depth * dir).y;
    if (dist <= 0.01f || depth > 100.f) {
        break;
    }
    // Move along the view ray
    depth += dist;
}
return vec2(depth, tag);
}

float opDisplace(vec3 p, vec3 eye )
{
    float d1 = dist(p, eye).x;
    float d2 = d1 * sin(20.f *p.x)*sin(20.f *p.y)*sin(20.f*p.z);
    return d1+d2;
}



mat4 viewMatrix(vec3 eye, vec3 center, vec3 up) {
	vec3 f = normalize(center - eye);
	vec3 s = normalize(cross(f, up));
	vec3 u = cross(s, f);
	return mat4(
		vec4(s, 0.0),
		vec4(u, 0.0),
		vec4(-f, 0.0),
		vec4(0.0, 0.0, 0.0, 1)
	);
}



void main() {

  //vec2 screen = vec2((fs_Pos.x / u_Dimensions.x) * 2.f - 1.f, 1.f - (fs_Pos.y / u_Dimensions.y) * 2.f);

//get ray marching direction 
  vec2 screen = fs_Pos.xy;
  vec3 right = cross(normalize(u_Ref - u_Eye), normalize(u_Up));
  float alpha = 50.f * 3.14159 / 180.0;
  float len = length(u_Ref - u_Eye);
  vec3 V = u_Up * len * tan(alpha);
  vec3 H = right * len * u_Dimensions.x / u_Dimensions.y * tan(alpha);
  vec3 P = u_Ref + screen.x * H + screen.y * V;
  vec3 dir = normalize(P - u_Eye);


//intersection
  float t = dist(dir, u_Eye).x;
  float tag = dist(dir, u_Eye).y;

  //float t = opDisplace(dir, u_Eye);
  vec3 isect = u_Eye + t * dir;

//phong (called normal function)
  vec3 V_P = normalize(u_Eye - P);
  vec3 L_P = normalize(vec3(-5.f) - P);
  vec3 H_P = (V_P + L_P) / 2.f;




  float spec = abs(pow(dot(H_P, normalize(normal(isect))), 1.f));

  float lambert = dot(normalize(-normal(isect)), vec3(-1.f));

//SDF
  vec3 color;
  if(t < 100.f)
  {
    //tag = 1.f;
    int tag_int = int(tag);
    switch(tag_int)
    {
      case 0: color = vec3(0.9137, 0.7176, 0.3569);break;
      case 1: color = vec3(0.7686, 0.8745, 0.9608);break;
      case 3: 
        float height = perlin_noise((vec2(isect.x + isect.y, isect.z + isect.y) ) / 4.f);
        //float height_forest = perlin_noise((vec2(fs_Pos.x, fs_Pos.z) ) / 3.f);
        float worley_Offset = worleyFBM(isect * 0.05);

        float fs_Height = height * 6.f;

        vec3 color1 = vec3(0.4392, 0.7686, 0.9882) ;
        vec3 color2 = vec3(0.7725, 0.8235, 0.8784) ;
        color = mix(color2, color1, worley_Offset);
        break;
      case 2: color = vec3(0.698, 0.5412, 0.9529);break;
      case 4: color = vec3(0.1451, 0.1137, 0.0118);break;
    //   case 7: vec3 color1 = vec3(0.3529, 0.2157, 0.0275);
    //           float height = mix(0.f, 1.f, isect.y);
    //           //float perlin = cnoise(vec3(height+ isect.y) * 0.3);
    //           float perlin = pnoise(vec3(isect), vec3(2.f));
    //           vec3 color2 = vec3(0.9059, 0.7216, 0.3765);
    //           color = mix(color2, color1, perlin);break;
    //   case 8: vec3 color3 = vec3(0.3529, 0.2157, 0.0275);
    //           float height1 = mix(0.f, 1.f, isect.z);
    //           float perlin1 = cnoise(vec3(height+ isect.z) * 0.3);
    //           vec3 color4 = vec3(0.9059, 0.7216, 0.3765);
    //           color = mix(color2, color1, perlin1);break;
    }

    //color = vec3(tag / 4.0);
     //color = vec3(0.8667, 0.5882, 0.0667);
     //color = pow( color, vec3(0.4545) );
     out_Col = vec4(color * (spec), 1.f);
  }
  else
  {
    //   if(fs_Pos.y < 0.05)
    //   {
    //     float height = perlin_noise((vec2(dir.x, dir.y) ) / 4.f);
    //     float height_forest = perlin_noise((vec2(fs_Pos.x, fs_Pos.z) ) / 3.f);
    //     float worley_Offset = worleyFBM(dir * 0.5 + u_Eye);

    //     float fs_Height = height * 6.f;

    //     vec3 color1 = vec3(0.7333, 0.8627, 0.9451) ;
    //     vec3 color2 = vec3(0.0196, 0.1922, 0.302) ;
    //     vec3 color = mix(color2, color1, worley_Offset * 2.0);
    //     //float terrain_y = smoothstep(tree_Height, fs_Height_Forest, fs_Pos.y);

    //     out_Col = vec4(vec3(color), 1.f);
      //}

        
  }


}

#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;
uniform vec4 u_Size;

in vec4 fs_Pos;
out vec4 out_Col;
#define iterations 17
#define formuparam 0.53

#define volsteps 20
#define stepsize 0.1

#define zoom   0.800
#define tile   0.850
#define speed  0.010 

#define brightness 0.0015
#define darkmatter 0.300
#define distfading 0.730
#define saturation 0.850

vec3 c_seed = vec3(0);
float PI_2 = 6.2831853;


// 3D Perlin Noise
/////////////////////////////////////////


float rand(vec2 uv) {
    const highp float a = 12.9898;
    const highp float b = 78.233;
    const highp float c = 43758.5453;
    highp float dt = dot(uv, vec2(a, b));
    highp float sn = mod(dt, 3.1415);
    return fract(sin(sn) * c);
}

float random1( vec3 p , vec3 seed) {
  return fract(sin(dot(p + seed, vec3(987.654, 123.456, 531.975))) * 85734.3545);
}

void draw_stars(inout vec4 color, vec2 uv) {
    float t = sin(u_Time * 0.1 * rand(-uv)) * 0.5 + 0.5;
    //color += step(0.99, stars) * t;
    color += smoothstep(0.99, 1.0, rand(uv)) * t;
}

vec3 randGrad(vec3 p, vec3 seed) {
  switch(int(floor(random1(p, seed) * 6.))) {
    case 0: return vec3(1., 0., 0.);
    case 1: return vec3(-1., 0., 0.);
    case 2: return vec3(0., 1., 0.);
    case 3: return vec3(0., -1., 0.);
    case 4: return vec3(0., 0., 1.);
    default: return vec3(0., 0., -1.);
  }
}


float falloff(float t) {
  t = t * t * t * (t * (t * 6. - 15.) + 10.);
  return t;
}

float PerlinNoise(vec3 p, float s) {
    p /= s;
    vec3 pCell = floor(p);
    p -= pCell;
    float dotGrad000 = dot(randGrad(pCell + vec3(0., 0., 0.), c_seed), p - vec3(0., 0., 0.));
    float dotGrad010 = dot(randGrad(pCell + vec3(0., 1., 0.), c_seed), p - vec3(0., 1., 0.));
    float dotGrad100 = dot(randGrad(pCell + vec3(1., 0., 0.), c_seed), p - vec3(1., 0., 0.));
    float dotGrad110 = dot(randGrad(pCell + vec3(1., 1., 0.), c_seed), p - vec3(1., 1., 0.));
    float dotGrad001 = dot(randGrad(pCell + vec3(0., 0., 1.), c_seed), p - vec3(0., 0., 1.));
    float dotGrad011 = dot(randGrad(pCell + vec3(0., 1., 1.), c_seed), p - vec3(0., 1., 1.));
    float dotGrad101 = dot(randGrad(pCell + vec3(1., 0., 1.), c_seed), p - vec3(1., 0., 1.));
    float dotGrad111 = dot(randGrad(pCell + vec3(1., 1., 1.), c_seed), p - vec3(1., 1., 1.));

    float mixedDGX00 = mix(dotGrad000, dotGrad100, falloff(p.x));
    float mixedDGX10 = mix(dotGrad010, dotGrad110, falloff(p.x));
    float mixedDGX01 = mix(dotGrad001, dotGrad101, falloff(p.x));
    float mixedDGX11 = mix(dotGrad011, dotGrad111, falloff(p.x));

    float mixedDGY0 = mix(mixedDGX00, mixedDGX10, falloff(p.y));
    float mixedDGY1 = mix(mixedDGX01, mixedDGX11, falloff(p.y));

    return mix(mixedDGY0, mixedDGY1, falloff(p.z)) * .5 + .5;
}


float FBMPerlin(vec3 p) {
    float sum = 0.;
    float noise = 0.;
    int maxIter = 4;
    float minCell = 2.;
    for (int i = 0; i < maxIter; i++) {
        noise += PerlinNoise(p, minCell * pow(2., float(i))) / pow(2., float(maxIter - i));
        sum += 1. / pow(2., float(maxIter - i));
    }
    noise /= sum;
    return noise;
}

float warpFBMPerlin(vec3 p, int time) {
  vec3 q = vec3(FBMPerlin(p + vec3(0. + 0.5 * float(u_Time),0., 0.)) + 0.003 * float(u_Time),
                FBMPerlin(p + vec3(3., 5., 2.)),
                FBMPerlin(p + vec3(2., -1., 1.)));
  return FBMPerlin(p + 30.0 * q);
}


//fbm
const mat2 m = mat2( 0.80,  0.60, -0.60,  0.80 );

float noise( in vec2 x )
{
	return sin(1.5*x.x)*sin(1.5*x.y);
}

float fbm4( vec2 p )
{
    float f = 0.0;
    f += 0.5000*noise( p ); p = m*p*2.02;
    f += 0.2500*noise( p ); p = m*p*2.03;
    f += 0.1250*noise( p ); p = m*p*2.01;
    f += 0.0625*noise( p );
    return f/0.9375;
}

float fbm6( vec2 p )
{
    float f = 0.0;
    f += 0.500000*(0.5+0.5*noise( p )); p = m*p*2.02;
    f += 0.250000*(0.5+0.5*noise( p )); p = m*p*2.03;
    f += 0.125000*(0.5+0.5*noise( p )); p = m*p*2.01;
    f += 0.062500*(0.5+0.5*noise( p )); p = m*p*2.04;
    f += 0.031250*(0.5+0.5*noise( p )); p = m*p*2.01;
    f += 0.015625*(0.5+0.5*noise( p ));
    return f/0.96875;
}


float func( vec2 q, out vec4 ron )
{
    float ql = length( q );
    q.x += 0.05*sin(0.01*u_Time+ql*4.1);
    q.y += 0.05*sin(0.01*u_Time+ql*4.3);
    q *= 0.5;

	vec2 o = vec2(0.0);
    o.x = 0.5 + 0.5*fbm4( vec2(2.0*q          )  );
    o.y = 0.5 + 0.5*fbm4( vec2(2.0*q+vec2(5.2))  );

	float ol = length( o );
    o.x += 0.02*sin(0.01*u_Time+ol)/ol;
    o.y += 0.02*sin(0.01*u_Time+ol)/ol;

    vec2 n;
    n.x = fbm6( vec2(4.0*o+vec2(9.2))  );
    n.y = fbm6( vec2(4.0*o+vec2(5.7))  );

    vec2 p = 4.0*q + 4.0*n;

    float f = 0.5 + 0.5*fbm4( p );

    f = mix( f, f*f*f*3.5, f*abs(n.x) );

    float g = 0.5 + 0.5*sin(4.0*p.x)*sin(4.0*p.y);
    f *= 1.0-0.5*pow( g, 8.0 );

	ron = vec4( o, n );
	
    return f;
}

float func1( vec2 q, out vec4 ron )
{
    float ql = length( q );
    // q.x += 0.05*sin(0.005*u_Time+ql*4.1);
    // q.y += 0.05*sin(0.005*u_Time+ql*4.3);
    q *= 0.5;

	vec2 o = vec2(0.0);
    o.x = 0.5 + 0.5*fbm4( vec2(2.0*q          )  );
    o.y = 0.5 + 0.5*fbm4( vec2(2.0*q+vec2(5.2))  );

	float ol = length( o );
    // o.x += 0.02*sin(0.005*u_Time+ol)/ol;
    // o.y += 0.02*sin(0.005*u_Time+ol)/ol;

    vec2 n;
    n.x = fbm6( vec2(4.0*o+vec2(9.2))  );
    n.y = fbm6( vec2(4.0*o+vec2(5.7))  );

    vec2 p = 4.0*q + 4.0*n;

    float f = 0.5 + 0.5*fbm4( p );

    f = mix( f, f*f*f*3.5, f*abs(n.x) );

    float g = 0.5 + 0.5*sin(4.0*p.x)*sin(4.0*p.y);
    f *= 1.0-0.5*pow( g, 8.0 );

	ron = vec4( o, n );
	
    return f;
}

float func2( vec2 q, out vec4 ron )
{
    float ql = length( q );
    // q.x += 0.05*sin(0.005*u_Time+ql*4.1);
    // q.y += 0.05*sin(0.005*u_Time+ql*4.3);
    q *= 0.5;

	vec2 o = vec2(0.0);
    o.x = 0.5 + 0.5*fbm4( vec2(2.0*q          )  );
    o.y = 0.5 + 0.5*fbm4( vec2(2.0*q+vec2(5.2))  );

	float ol = length( o );
    // o.x += 0.02*sin(0.005*u_Time+ol)/ol;
    // o.y += 0.02*sin(0.005*u_Time+ol)/ol;

    vec2 n;
    n.x = fbm4( vec2(4.0*o+vec2(9.2))  );
    n.y = fbm4( vec2(4.0*o+vec2(5.7))  );

    vec2 p = 4.0*q + 4.0*n;

    float f = 0.5 + 0.5*fbm4( p );

    f = mix( f, f*f*f*3.5, f*abs(n.x) );

    float g = 0.5 + 0.5*sin(4.0*p.x)*sin(4.0*p.y);
    f *= 1.0-0.5*pow( g, 8.0 );

	ron = vec4( o, n );
	
    return f;
}


vec3 doMagic(vec2 p)
{
	vec2 q = p*0.6;

    vec4 on = vec4(0.0);
    float f = func(q, on);

	vec3 col = vec3(0.0);
    col = mix( vec3(0.0196, 0.5804, 0.1608), vec3(0.0235, 0.2588, 0.4784), f );
    col = mix( col, vec3(0.1922, 0.0039, 0.1098), dot(on.zw,on.zw) );
    col = mix( col, vec3(0.502, 0.0118, 0.2157), 0.5*on.y*on.y );
    col = mix( col, vec3(0.0,0.2,0.4), 0.5*smoothstep(1.2,1.3,abs(on.z)+abs(on.w)) );
    col = clamp( col*f*2.0, 0.0, 1.0 );
	return 1.1*col*col;
}

vec3 doMagic1(vec2 p)
{
	vec2 q = p*0.6;

    vec4 on = vec4(0.0);
    float f = func2(q, on);

	vec3 col = vec3(1.0, 1.0, 1.0);
    col = mix( vec3(0.349, 0.7412, 0.9686), vec3(1.0, 1.0, 1.0), f );
    col = mix( col, vec3(0.4, 0.8627, 0.9451), 0.5*smoothstep(1.2,1.3,abs(on.z)) );
   //col = clamp( col*f*2.0, 0.0, 1.0 );
	return 1.5*col;
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
    float step = smoothstep(-u_Size.w * 0.05, u_Size.w * 0.05, 0.01 * sin(u_Time * 0.01));
    float c = sin(step);
    float s = sin(step);
    mat2  m = mat2(c,-s,s,c);
    vec2 r = m * p.yz;
    vec3  q = vec3(p.x, r.x, r.y);
    return q;
}

float onion( in float d, in float h )
{
    return abs(d)-h;
}


float sphSoftShadow( in vec3 ro, in vec3 rd, in vec4 sph, in float k )
{
    vec3 oc = ro - sph.xyz;
    float b = dot( oc, rd );
    float c = dot( oc, oc ) - sph.w*sph.w;
    float h = b*b - c;
    

    // physically plausible shadow
    float d = sqrt( max(0.0,sph.w*sph.w-h)) - sph.w;
    float t = -b - sqrt( max(h,0.0) );
    return (t<0.0) ? 1.0 : smoothstep(0.0, 1.0, 2.5*k*d/t ); 
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
    float k = abs(dot(q,vec2(-b,a)));
    
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



vec2 flame(vec3 pos)
{
    pos -= vec3(0.0, 35.0, 0.0);   
    pos /= 3.0;    
    float d = 1e10;
    float d1 = sdCylinder(pos, vec2(1.6, 10.f));
    
    //d1 *= radius;

    d = min(d, d1);

    return vec2(d * 3.f, 0.f);
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
    //float d_invisible = sphere(pos - vec3(0.f, 1.8, 0.f), 2.0f);
    d1 = d1 - 0.2 * sin(10.f * pos.x)*sin(8.f * pos.y)*sin(6.f * pos.z);
    float d2 = sdCylinder(pos + vec3(0.f, 0.5f, 0.f), vec2(2.f, 2.5f));
    vec3 box_pos = opRep(vec3(atan(pos.x, pos.z), pos.y, 0.5 * length(pos)) , vec3(1.0, 0.0, 1.0));

   
    float d3 = sdBox(box_pos, vec3(0.2, 2.0, 0.1));

    //float d3 = sdBox(pos, vec3(0.2, 2.0, 0.1));

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


vec2 map(vec3 pos)
{
    float d = 1e10;
   

    float tag = 1.f;
    
    float d1 = obj1(pos).x;
   float d2 = obj2(pos).x;
   float d3 = obj3(pos).x;

    d1 = min(d,  opSmoothSubtraction(d2, d1, 0.1));
    d1 = min(d, opSmoothUnion(d1, d3, 0.1));

    //d1 = min(d1, opSmoothUnion(d1, d4, 0.1));

    if(d1 < d)
    {
        d = d1;
        tag = obj1(pos).y;
    }

    float d4 = flame(pos).x;
    d4 = opSmoothUnion(d, d4, 0.1);
    if(d4 < d)
    {
        d = d4;
        tag = 0.f;
    }

    vec2 p = 0.01 * vec2(pos.x + pos.y, pos.y + pos.z);
    vec4 on = vec4(0.0);

    vec3 height = doMagic1(p);

    float d5 = obj5(pos+ vec3(0.0, 11.0 + length(height) * 2.0, 0.0)).x;

    d5 = opSmoothUnion(d, d5, 0.1);
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


void main() {

  //vec2 screen = vec2((fs_Pos.x / u_Dimensions.x) * 2.f - 1.f, 1.f - (fs_Pos.y / u_Dimensions.y) * 2.f);

  vec3 look = vec3(u_Eye.x, u_Eye.y, -u_Eye.z * 3.f);

//get ray marching direction 
  vec2 screen = fs_Pos.xy;
  vec3 right = cross(normalize(u_Ref - look), normalize(u_Up));
  float alpha = 50.f * 3.14159 / 180.0;
  float len = length(u_Ref - look);
  vec3 V = u_Up * len * tan(alpha);
  vec3 H = right * len * u_Dimensions.x / u_Dimensions.y * tan(alpha);
  vec3 P = u_Ref + screen.x * H + screen.y * V;
  vec3 dir = normalize(P - look);


//intersection
  float t = dist(dir, look).x;
  float tag = dist(dir, look).y;

  //float t = opDisplace(dir, u_Eye);
  vec3 isect = look + t * dir;

  //lights     
    vec4 lights[3];
    vec3 lightColor[3];
    lights[0] = vec4(16.0, 13.0, 15.0, 3.0); // key light
    lights[1] = vec4(-16.0, 13.0, 15.0, 2.0); // fill light
	lights[2] = vec4(0.0, 50.0, 0.0, 1.0); // back light
    //lights[3] = vec4(0.0, -15.0, 0.0, 10.0);
    
    
    lightColor[0] =vec3(0.4706, 0.3255, 0.7412); 
    lightColor[1] = vec3(0.4, 0.7, 1.0);
    lightColor[2] = vec3(0.6941, 0.8824, 0.9922);
    //lightColor[3] = vec3(1.0);


     //phong (called normal function)
    vec3 V_P = normalize(look - P);
    vec3 L_P = normalize(vec3(-5.f) - P);
    vec3 H_P = (V_P + L_P) / 2.f;

    vec3 color = vec3(0.0);


    float spec = abs(pow(dot(H_P, normalize(normal(isect))), 2.f));
    //float lambert = dot(normalize(-normal(isect)), vec3(-1.f));

    vec3 sum = vec3(0.f);


   

//SDF
  if(t < 100.f)
  {
    //tag = 1.f;
    int tag_int = int(tag);
    switch(tag_int)
    {
    case 0:
    vec3 color1 = vec3(0.102, 0.6431, 0.8941);
    vec3 color2 = vec3(0.0, 0.0, 0.0);
           float fog_noise = smoothstep(0.3, 0.6, warpFBMPerlin(vec3(isect.x + fs_Pos.x, fs_Pos.y * 40., isect.y + fs_Pos.z), int(u_Time * 2.0))) * pow((1. - fs_Pos.y), 2.);
       color = mix(color1, color2, fog_noise);

    //color = mix(color1, color2, abs(sin(-isect.y * 0.1 + u_Time * 0.1)));

    //color = color1;
    break;

      case 1: 
        color += vec3(0.9725, 0.9804, 0.9804);
        break;
      case 3: 
        vec2 p = 0.01* vec2(isect.x + isect.y, isect.y + isect.z);
        color += doMagic1(vec2(p));
        break;
    }


    vec3 nor = normalize(normal(isect));

        for (int i = 0; i < 3; i++) 
        { // Compute lambert for each light
            vec3 lambert = color * min(max(dot(nor, normalize(lights[i].xyz - isect)), 0.0f), 1.0f) * lights[i].w * lightColor[i];

            //vec4 sph1 = vec4(0.0, 10.0, 0.0, 15.0);
            sum += lambert * sphSoftShadow( isect, normalize(lights[i].xyz - isect), vec4(0.0,0.,0., 10.f), 16.0 );     
        }  

        if(tag_int != 0)
            color += sum / 3.0;  


    float ambient = 0.4;

    if(tag_int == 0)
    {

        out_Col = vec4(color, 1.0);        
    }
    else
    {

        if(tag_int == 1)
        {
            out_Col = vec4(color * spec, 1.0);            
        }
        else
        {
            out_Col = vec4(color *  ambient, 1.f);            
        }
    }
  }
  else
  {
     vec2 p = (-u_Dimensions.xy+200.0*vec2(fs_Pos))/u_Dimensions.y;

    vec2 uv = isect.xy / u_Dimensions.xy;
    vec3 color = doMagic(p);
    out_Col = vec4(color * 1.1, 1.0 );
    draw_stars(out_Col, uv);
    
  }


}

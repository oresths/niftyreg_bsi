#ifndef CUDALIKE_MATH_H_
#define CUDALIKE_MATH_H_

struct float3_t
{
    float x, y, z;
};
struct float4_t
{
    float x, y, z, w;
};
struct double3_t
{
    double x, y, z;
};
struct double4_t
{
    double x, y, z, w;
};
struct int3_t
{
    int x, y, z;
};

static int3_t make_int3_t(int x, int y, int z)
{
  int3_t t; t.x = x; t.y = y; t.z = z; return t;
}
static float3_t make_float3_t(float x, float y, float z)
{
  float3_t t; t.x = x; t.y = y; t.z = z; return t;
}
static float4_t make_float4_t(float x, float y, float z, float w)
{
  float4_t t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}
static double3_t make_double3_t(double x, double y, double z)
{
  double3_t t; t.x = x; t.y = y; t.z = z; return t;
}
static double4_t make_double4_t(double x, double y, double z, double w)
{
  double4_t t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

static float3_t operator*(float a, float3_t b)
{
    return make_float3_t(a * b.x, a * b.y, a * b.z);
}
static float3_t operator*(float3_t a, float b)
{
    return make_float3_t(a.x * b, a.y * b, a.z * b);
}
static float3_t operator*(float3_t a, float3_t b)
{
    return make_float3_t(a.x * b.x, a.y * b.y, a.z * b.z);
}
static float3_t operator-(float a, float3_t b)
{
    return make_float3_t(a - b.x, a - b.y, a - b.z);
}
static float3_t operator-(float3_t a, float b)
{
    return make_float3_t(a.x - b, a.y - b, a.z - b);
}
static float3_t operator-(float3_t a, float3_t b)
{
    return make_float3_t(a.x - b.x, a.y - b.y, a.z - b.z);
}
static float3_t operator+(float a, float3_t b)
{
    return make_float3_t(a + b.x, a + b.y, a + b.z);
}
static float3_t operator+(float3_t a, float b)
{
    return make_float3_t(a.x + b, a.y + b, a.z + b);
}
static float3_t operator+(float3_t a, float3_t b)
{
    return make_float3_t(a.x + b.x, a.y + b.y, a.z + b.z);
}
static float3_t operator/(float a, float3_t b)
{
    return make_float3_t(a / b.x, a / b.y, a / b.z);
}
static float3_t operator/(float3_t a, float b)
{
    return make_float3_t(a.x / b, a.y / b, a.z / b);
}
static float3_t operator/(float3_t a, float3_t b)
{
    return make_float3_t(a.x / b.x, a.y / b.y, a.z / b.z);
}

static float4_t operator*(float a, float4_t b)
{
    return make_float4_t(a * b.x, a * b.y, a * b.z, a * b.w);
}
static float4_t operator*(float4_t a, float b)
{
    return make_float4_t(a.x * b, a.y * b, a.z * b, a.w * b);
}
static float4_t operator*(float4_t a, float4_t b)
{
    return make_float4_t(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
static float4_t operator-(float a, float4_t b)
{
    return make_float4_t(a - b.x, a - b.y, a - b.z, a - b.w);
}
static float4_t operator-(float4_t a, float b)
{
    return make_float4_t(a.x - b, a.y - b, a.z - b, a.w - b);
}
static float4_t operator-(float4_t a, float4_t b)
{
    return make_float4_t(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
static float4_t operator+(float a, float4_t b)
{
    return make_float4_t(a + b.x, a + b.y, a + b.z, a + b.w);
}
static float4_t operator+(float4_t a, float b)
{
    return make_float4_t(a.x + b, a.y + b, a.z + b, a.w + b);
}
static float4_t operator+(float4_t a, float4_t b)
{
    return make_float4_t(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
static float4_t operator/(float a, float4_t b)
{
    return make_float4_t(a / b.x, a / b.y, a / b.z, a / b.w);
}
static float4_t operator/(float4_t a, float b)
{
    return make_float4_t(a.x / b, a.y / b, a.z / b, a.w / b);
}
static float4_t operator/(float4_t a, float4_t b)
{
    return make_float4_t(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

static double3_t operator*(double a, double3_t b)
{
    return make_double3_t(a * b.x, a * b.y, a * b.z);
}
static double3_t operator*(double3_t a, double b)
{
    return make_double3_t(a.x * b, a.y * b, a.z * b);
}
static double3_t operator*(double3_t a, double3_t b)
{
    return make_double3_t(a.x * b.x, a.y * b.y, a.z * b.z);
}
static double3_t operator-(double a, double3_t b)
{
    return make_double3_t(a - b.x, a - b.y, a - b.z);
}
static double3_t operator-(double3_t a, double b)
{
    return make_double3_t(a.x - b, a.y - b, a.z - b);
}
static double3_t operator-(double3_t a, double3_t b)
{
    return make_double3_t(a.x - b.x, a.y - b.y, a.z - b.z);
}
static double3_t operator+(double a, double3_t b)
{
    return make_double3_t(a + b.x, a + b.y, a + b.z);
}
static double3_t operator+(double3_t a, double b)
{
    return make_double3_t(a.x + b, a.y + b, a.z + b);
}
static double3_t operator+(double3_t a, double3_t b)
{
    return make_double3_t(a.x + b.x, a.y + b.y, a.z + b.z);
}
static double3_t operator/(double a, double3_t b)
{
    return make_double3_t(a / b.x, a / b.y, a / b.z);
}
static double3_t operator/(double3_t a, double b)
{
    return make_double3_t(a.x / b, a.y / b, a.z / b);
}
static double3_t operator/(double3_t a, double3_t b)
{
    return make_double3_t(a.x / b.x, a.y / b.y, a.z / b.z);
}

static double4_t operator*(double a, double4_t b)
{
    return make_double4_t(a * b.x, a * b.y, a * b.z, a * b.w);
}
static double4_t operator*(double4_t a, double b)
{
    return make_double4_t(a.x * b, a.y * b, a.z * b, a.w * b);
}
static double4_t operator*(double4_t a, double4_t b)
{
    return make_double4_t(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
static double4_t operator-(double a, double4_t b)
{
    return make_double4_t(a - b.x, a - b.y, a - b.z, a - b.w);
}
static double4_t operator-(double4_t a, double b)
{
    return make_double4_t(a.x - b, a.y - b, a.z - b, a.w - b);
}
static double4_t operator-(double4_t a, double4_t b)
{
    return make_double4_t(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
static double4_t operator+(double a, double4_t b)
{
    return make_double4_t(a + b.x, a + b.y, a + b.z, a + b.w);
}
static double4_t operator+(double4_t a, double b)
{
    return make_double4_t(a.x + b, a.y + b, a.z + b, a.w + b);
}
static double4_t operator+(double4_t a, double4_t b)
{
    return make_double4_t(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
static double4_t operator/(double a, double4_t b)
{
    return make_double4_t(a / b.x, a / b.y, a / b.z, a / b.w);
}
static double4_t operator/(double4_t a, double b)
{
    return make_double4_t(a.x / b, a.y / b, a.z / b, a.w / b);
}
static double4_t operator/(double4_t a, double4_t b)
{
    return make_double4_t(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

#endif /* CUDALIKE_MATH_H_ */


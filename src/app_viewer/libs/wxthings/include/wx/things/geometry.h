///////////////////////////////////////////////////////////////////////////////
// Name:        geometry.h
// Purpose:     Additional geometry functions for wxWidgets (see wx/geometry.h)
// Author:      John Labenski
// Created:     07/01/02
// Copyright:   John Labenski, 2002
// License:     wxWidgets v2
///////////////////////////////////////////////////////////////////////////////

#ifndef __WXIMAGER_GEOMETRY_H__
#define __WXIMAGER_GEOMETRY_H__

#include "wx/geometry.h"
#include "wx/things/thingdef.h"

#define wxGEOMETRY_INF 1E100

//-----------------------------------------------------------------------------
// wxRay2DDouble uses point slope line format
//
//    y = mx+b, m=(x-x0)/(y-y0)
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_THINGS wxRay2DDouble : public wxPoint2DDouble {
  public:
    inline wxRay2DDouble(wxDouble x = 0, wxDouble y = 0, wxDouble slope = 0) {
        _x = x;
        _y = y;
        _slope = slope;
    }

    inline wxRay2DDouble(const wxPoint2DDouble& pt, wxDouble slope) {
        _x = pt._x;
        _y = pt._y;
        _slope = slope;
    }

    inline wxRay2DDouble(wxDouble x1, wxDouble y1, wxDouble x2, wxDouble y2) {
        _x = x1;
        _y = y1;
        _slope = GetSlope(x1, y1, x2, y2);
    }

    inline wxRay2DDouble(const wxPoint2DDouble& pt1, const wxPoint2DDouble& pt2) {
        _x = pt1._x;
        _y = pt1._y;
        _slope = GetSlope(pt1, pt2);
    }

    inline wxRay2DDouble(const wxRay2DDouble& line) {
        _x = line._x;
        _y = line._y;
        _slope = line._slope;
    }

    inline wxDouble GetX() const {
        return _x;
    }

    inline wxDouble GetY() const {
        return _y;
    }

    inline wxPoint2DDouble GetPoint() const {
        return (wxPoint2DDouble)(*this);
    }

    inline wxDouble GetSlope() const {
        return _slope;
    }

    inline void SetX(wxDouble x) {
        _x = x;
    }

    inline void SetY(wxDouble y) {
        _y = y;
    }

    inline void SetPoint(const wxPoint2DDouble& pt) {
        _x = pt._x;
        _y = pt._y;
    }

    inline void SetSlope(wxDouble slope) {
        _slope = slope;
    }

    inline wxDouble GetYFromX(wxDouble x) const {
        return _slope * (x - _x) + _y;
    }

    inline wxDouble GetXFromY(wxDouble y) const {
        return (y - _y) / _slope + _x;
    }

    // Get a point along the line at pos x or y
    inline wxPoint2DDouble GetPointOnRayFromX(wxDouble x) const {
        return wxPoint2DDouble(x, GetYFromX(x));
    }

    inline wxPoint2DDouble GetPointOnRayFromY(wxDouble y) const {
        if (_slope == 0) return (*this);
        return wxPoint2DDouble(GetXFromY(y), y);
    }

    // Translate the point _pt along the line to pos x or y
    inline void TranslatePointByX(wxDouble x) {
        _y = GetYFromX(x);
        _x = x;
    }

    inline void TranslatePointByY(wxDouble y) {
        _x = GetXFromY(y);
        _y = y;
    }

    inline wxRay2DDouble GetTranslatedLineByX(wxDouble x) const {
        return wxRay2DDouble(x, GetYFromX(x), _slope);
    }

    inline wxRay2DDouble GetTranslatedLineByY(wxDouble y) const {
        return wxRay2DDouble(GetXFromY(y), y, _slope);
    }

    inline wxDouble GetDistanceToPoint(const wxPoint2DDouble& pt, wxPoint2DDouble* closestPt = NULL) const {
        wxPoint2DDouble l1(_x, _y);
        wxPoint2DDouble l2(GetPointOnRayFromX(_x + pt._x));
        wxPoint2DDouble v = l2 - l1;
        wxPoint2DDouble w = pt - l1;
        wxDouble c1 = w.GetDotProduct(v);
        wxDouble c2 = v.GetDotProduct(v);
        wxDouble b = c1 / c2;
        wxPoint2DDouble pb = l1 + b * v;
        if (closestPt) *closestPt = pb;
        return pb.GetDistance(pt);

        /*
                wxPoint2DDouble ll2 = l2;
                double l = ll2.GetDistanceSquare(l1);
                double u = ((p._x-l1._x)*(l2._x-l1._x)+(p._y-l1._y)*(l2._y-l1._y))/l;
                wxPoint2DDouble i = l1 + u*(l2-l1);

                //printf("distance %.9lf %.9lf %d\n\n", i.GetDistance(p), pb.GetDistance(p), int(pb==i));
        */

        //                   this x for y on this line
        // y(on this line) = _slope*(x-_pt._x) + _pt._y
        //                 = (-1/_slope)(x-pt._x) + pt._y
        // so
        // x(on this line) = (y-_pt._y)/_slope + _pt._x
        //                 = (y-pt._y)/(-1/_slope) + _pt._x

        //(y-_pt._y)/_slope + _pt._x = (y-pt._y)(-1/_slope) + _pt._x
        //(y-_pt._y)/_slope - _slope*(y-pt._y) = + _pt._x - _pt._x
        // y*(1/_slope - _slope) = + _pt._x - _pt._x +_slope*pt._y + _slope*_pt._y
        // y = (_pt._x - _pt._x +_slope*pt._y + _slope*_pt._y)/(1/_slope - _slope)
        /*
                wxDouble x = (_x + _slope*_slope*_x - _slope*(_y - _y))/(_slope*_slope+1.0);
                wxPoint2DDouble pl(x, _slope*(x-_x) + _y);
                if (closestPt) *closestPt = pl;
                return pl.GetDistance(pt);
        */
    }

    inline wxDouble GetDistanceToRay(const wxRay2DDouble& ray) const {
        // FIXME - unchecked, just quickly translated from some other code
        if (_slope != ray._slope) return 0;
        if (_slope == 0) return fabs(_y - ray._y);
        wxPoint2DDouble p1 = GetPointOnRayFromX(0);
        wxPoint2DDouble p2 = ray.GetPointOnRayFromX(0);
        // y = (-1/s1)*x+p1._y = s2*x + p2._y
        wxDouble dx = (p1._y - p2._y) / (_slope + (1.0 / _slope));
        wxDouble dy = (_slope * dx + p2._y) - p1._y;
        return sqrt(dx * dx + dy * dy);
    }

    inline static wxDouble GetSlope(wxDouble x1, wxDouble y1, wxDouble x2, wxDouble y2) {
        return (y2 - y1) / (x2 - x1);
    }

    inline static wxDouble GetSlope(const wxPoint2DDouble& pt1, const wxPoint2DDouble& pt2) {
        return (pt2._y - pt1._y) / (pt2._x - pt1._x);
    }

    // find the point where the two rays meet, return false if parallel
    bool Intersect(const wxRay2DDouble& other, wxPoint2DDouble& pt) const {
        // (y1-y0)/(x1-x0)=m for both lines, equate y1's first
        if (_slope == other._slope) return false;
        pt._x = (_slope * _x - other._slope * other._x + other._y - _y) / (_slope - other._slope);
        pt._y = GetYFromX(pt._x);
        return true;
    }

    // Operators

    inline wxRay2DDouble operator=(const wxRay2DDouble& r) {
        _x = r._x;
        _y = r._y;
        _slope = r._slope;
        return *this;
    }

    inline bool operator==(const wxRay2DDouble& r) const {
        return (_x == r._x) && (_y == r._y) && (_slope == r._slope);
    }

    inline bool operator!=(const wxRay2DDouble& r) const {
        return !(*this == r);
    }

    inline wxRay2DDouble operator+(const wxPoint2DDouble& rel_pos) const {
        return wxRay2DDouble(_x + rel_pos._x, _y + rel_pos._y, _slope);
    }

    inline wxRay2DDouble operator-(const wxPoint2DDouble& rel_pos) const {
        return wxRay2DDouble(_x - rel_pos._x, _y - rel_pos._y, _slope);
    }

    inline wxRay2DDouble operator*(const wxPoint2DDouble& rel_pos) const {
        return wxRay2DDouble(_x * rel_pos._x, _y * rel_pos._y, _slope);
    }

    inline wxRay2DDouble operator/(const wxPoint2DDouble& rel_pos) const {
        return wxRay2DDouble(_x / rel_pos._x, _y / rel_pos._y, _slope);
    }

    inline wxRay2DDouble& operator+=(const wxPoint2DDouble& rel_pos) {
        _x += rel_pos._x;
        _y += rel_pos._y;
        return *this;
    }

    inline wxRay2DDouble& operator-=(const wxPoint2DDouble& rel_pos) {
        _x -= rel_pos._x;
        _y -= rel_pos._y;
        return *this;
    }

    inline wxRay2DDouble& operator*=(const wxPoint2DDouble& rel_pos) {
        _x *= rel_pos._x;
        _y *= rel_pos._y;
        return *this;
    }

    inline wxRay2DDouble& operator/=(const wxPoint2DDouble& rel_pos) {
        _x /= rel_pos._x;
        _y /= rel_pos._y;
        return *this;
    }

    inline wxRay2DDouble operator+(const wxDouble& rel_slope) const {
        return wxRay2DDouble(_x, _y, _slope + rel_slope);
    }

    inline wxRay2DDouble operator-(const wxDouble& rel_slope) const {
        return wxRay2DDouble(_x, _y, _slope - rel_slope);
    }

    inline wxRay2DDouble operator*(const wxDouble& rel_slope) const {
        return wxRay2DDouble(_x, _y, _slope * rel_slope);
    }

    inline wxRay2DDouble operator/(const wxDouble& rel_slope) const {
        return wxRay2DDouble(_x, _y, _slope / rel_slope);
    }

    inline wxRay2DDouble& operator+=(const wxDouble& rel_slope) {
        _slope += rel_slope;
        return *this;
    }

    inline wxRay2DDouble& operator-=(const wxDouble& rel_slope) {
        _slope -= rel_slope;
        return *this;
    }

    inline wxRay2DDouble& operator*=(const wxDouble& rel_slope) {
        _slope *= rel_slope;
        return *this;
    }

    inline wxRay2DDouble& operator/=(const wxDouble& rel_slope) {
        _slope /= rel_slope;
        return *this;
    }

    wxDouble _slope;
};

//-----------------------------------------------------------------------------
// wxLine2DInt uses point slope line format
//
//    y = mx+b, m=(x-x0)/(y-y0)
//-----------------------------------------------------------------------------
/*
class WXDLLIMPEXP_THINGS wxLine2DInt : wxRect2DInt
{
public :

    inline wxLine2DInt(wxInt32 x1=0, wxInt32 y1=0, wxInt32 x2=0, wxInt32 y2=0)
        { _x = x1; _y = y1; _width = x2 - x1; _height = y2 - y1; }
    inline wxLine2DInt(const wxPoint2DInt &pt1, const wxPoint2DInt &pt2)
        { _x = pt1._x; _y = pt1._y; _width = pt2._x - pt1._x; _height = pt2._y - pt1._y; }
    inline wxLine2DInt(const wxLine2DInt &line)
        { _x = line._x; _y = line._y; _width = line._width; _height = line._height; }

    inline wxInt32 GetX1() const { return _x; }
    inline wxInt32 GetY1() const { return _y; }
    inline wxInt32 GetX2() const { return _x + _width; }
    inline wxInt32 GetY2() const { return _y + _height; }
    inline wxPoint2DInt Get1Point() const { return GetTopLeft(); }
    inline wxPoint2DInt Get2Point() const { return GetBottomRight(); }
    inline wxDouble GetSlope() const { return wxDouble(_height)/wxDouble(_width); }

    inline void SetX1(wxInt32 x) { _x = x; }
    inline void SetY1(wxInt32 y) { _y = y; }
    inline void SetX2(wxInt32 x) { _width = _x - x; }
    inline void SetY2(wxInt32 y) { _height = _y - y; }
    inline void SetPoint1(const wxPoint2DInt &pt ) { SetTopLeft(pt); }
    inline void SetPoint2(const wxPoint2DInt &pt ) { SetBottomRight(pt); }

    inline wxDouble GetYFromX(wxDouble x) const {return (wxDouble(_height)/_width)*(x-_x) + _y;}
    inline wxDouble GetXFromY(wxDouble y) const {return (y-_y)*(wxDouble(_height)/_width) + _x;}

    // Get a point along the line at pos x or y
    inline wxPoint2DInt GetPointOnLineFromX(wxInt32 x) const
        { return wxPoint2DInt(x, GetYFromX(x)); }
    inline wxPoint2DInt GetPointOnLineFromY(wxInt32 y) const
        { return wxPoint2DInt(GetXFromY(y), y); }

    // Translate the point _pt along the line to pos x or y
    inline void TranslatePointByX(wxDouble x)
        { _pt._y = GetYFromX(x); _pt._x = x; }
    inline void TranslatePointByY(wxDouble y)
        { _pt._x = GetXFromY(y); _pt._y = y; }
    inline wxLine2DInt GetTranslatedLineByX(wxDouble x)
        { return wxLine2DInt(x, GetYFromX(x), _slope); }
    inline wxLine2DInt GetTranslatedLineByY(wxDouble y)
        { return wxLine2DInt(GetXFromY(y), y, _slope); }

    inline wxDouble GetDistanceToPoint(const wxPoint2DDouble &pt, wxPoint2DDouble *closestPt=NULL) const
    {
        wxPoint2DDouble l1(_pt);
        wxPoint2DDouble l2(GetPointOnLineFromX(_pt._x+pt._x));
        wxPoint2DDouble v = l2 - l1;
        wxPoint2DDouble w = pt - l1;
        double c1 = w.GetDotProduct(v);
        double c2 = v.GetDotProduct(v);
        double b = c1 / c2;
        wxPoint2DDouble pb = l1 + b*v;
        if (closestPt) *closestPt = pb;
        return pb.GetDistance( pt );

    }


    inline static wxDouble GetSlope(wxDouble x1, wxDouble y1, wxDouble x2, wxDouble y2)
        { return (y2 - y1)/(x2 - x1); }
    inline static wxDouble GetSlope(const wxPoint2DDouble &pt1, const wxPoint2DDouble &pt2)
        { return (pt2._y-pt1._y)/(pt2._x-pt1._x); }


    // Default copy operator is ok

    wxPoint2DDouble _pt;
    wxDouble _slope;
};
*/

//-----------------------------------------------------------------------------
// wxCircleDouble   _r*_r = (x-_origin._x)^2 + (y-_origin._y)^2
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_THINGS wxCircleDouble : public wxPoint2DDouble {
  public:
    inline wxCircleDouble(const wxCircleDouble& circle) {
        _x = circle._x;
        _y = circle._y;
        _r = circle._r;
    }

    inline wxCircleDouble(wxDouble x = 0, wxDouble y = 0, wxDouble r = 0) {
        _x = x;
        _y = y;
        _r = r;
    }

    inline wxCircleDouble(const wxPoint2DDouble& origin, wxDouble r) {
        _x = origin._x;
        _y = origin._y;
        _r = r;
    }

    inline wxCircleDouble(const wxPoint2DDouble& p1, const wxPoint2DDouble& p2, const wxPoint2DDouble& p3);

    inline bool IsEmpty() const {
        return _r <= 0;
    }

    inline wxDouble GetX() const {
        return _x;
    }

    inline wxDouble GetY() const {
        return _y;
    }

    inline wxPoint2DDouble GetOrigin() const {
        return wxPoint2DDouble(_x, _y);
    }

    inline wxDouble GetRadius() const {
        return _r;
    }

    // Get a bounding rect
    inline wxRect2DDouble GetRect() const {
        return wxRect2DDouble(_x - _r, _y - _r, 2.0 * _r, 2.0 * _r);
    }

    inline void SetX(wxDouble x) {
        _x = x;
    }

    inline void SetY(wxDouble y) {
        _y = y;
    }

    inline void SetOrigin(const wxPoint2DDouble& origin) {
        _x = origin._x;
        _y = origin._y;
    }

    inline void SetRadius(wxDouble r) {
        _r = r;
    }

    inline bool GetPointInCircle(wxDouble x, wxDouble y) const {
        if (IsEmpty()) return false;
        return ((x - _x) * (x - _x) + (y - _y) * (y - _y) <= _r * _r);
    }

    inline bool GetPointInCircle(const wxPoint2DDouble& pt) const {
        return GetPointInCircle(pt._x, pt._y);
    }

    inline bool Intersects(const wxCircleDouble& circle) const {
        return GetDistance(circle) <= _r + circle._r;
    }

    int IntersectLine(const wxRay2DDouble& line, wxPoint2DDouble* pt1 = NULL, wxPoint2DDouble* pt2 = NULL) const;

    // Operators

    inline wxCircleDouble operator=(const wxCircleDouble& c) {
        _x = c._x;
        _y = c._y;
        _r = c._r;
        return *this;
    }

    inline bool operator==(const wxCircleDouble& c) const {
        return (_x == c._x) && (_y == c._y) && (_r == c._r);
    }

    inline bool operator!=(const wxCircleDouble& c) const {
        return !(*this == c);
    }

    inline wxCircleDouble operator+(const wxPoint2DDouble& rel_origin) const {
        return wxCircleDouble(_x + rel_origin._x, _y + rel_origin._y, _r);
    }

    inline wxCircleDouble operator-(const wxPoint2DDouble& rel_origin) const {
        return wxCircleDouble(_x - rel_origin._x, _y - rel_origin._y, _r);
    }

    inline wxCircleDouble operator*(const wxPoint2DDouble& rel_origin) const {
        return wxCircleDouble(_x * rel_origin._x, _y * rel_origin._y, _r);
    }

    inline wxCircleDouble operator/(const wxPoint2DDouble& rel_origin) const {
        return wxCircleDouble(_x / rel_origin._x, _y / rel_origin._y, _r);
    }

    inline wxCircleDouble& operator+=(const wxPoint2DDouble& rel_origin) {
        _x += rel_origin._x;
        _y += rel_origin._y;
        return *this;
    }

    inline wxCircleDouble& operator-=(const wxPoint2DDouble& rel_origin) {
        _x -= rel_origin._x;
        _y -= rel_origin._y;
        return *this;
    }

    inline wxCircleDouble& operator*=(const wxPoint2DDouble& rel_origin) {
        _x *= rel_origin._x;
        _y *= rel_origin._y;
        return *this;
    }

    inline wxCircleDouble& operator/=(const wxPoint2DDouble& rel_origin) {
        _x /= rel_origin._x;
        _y /= rel_origin._y;
        return *this;
    }

    inline wxCircleDouble operator+(const wxDouble& rel_radius) const {
        return wxCircleDouble(_x, _y, _r + rel_radius);
    }

    inline wxCircleDouble operator-(const wxDouble& rel_radius) const {
        return wxCircleDouble(_x, _y, _r - rel_radius);
    }

    inline wxCircleDouble operator*(const wxDouble& rel_radius) const {
        return wxCircleDouble(_x, _y, _r * rel_radius);
    }

    inline wxCircleDouble operator/(const wxDouble& rel_radius) const {
        return wxCircleDouble(_x, _y, _r / rel_radius);
    }

    inline wxCircleDouble& operator+=(const wxDouble& rel_radius) {
        _r += rel_radius;
        return *this;
    }

    inline wxCircleDouble& operator-=(const wxDouble& rel_radius) {
        _r -= rel_radius;
        return *this;
    }

    inline wxCircleDouble& operator*=(const wxDouble& rel_radius) {
        _r *= rel_radius;
        return *this;
    }

    inline wxCircleDouble& operator/=(const wxDouble& rel_radius) {
        _r /= rel_radius;
        return *this;
    }

    wxDouble _r;
};

//-----------------------------------------------------------------------------
// wxCircleInt   _r*_r = (x-_origin._x)^2 + (y-_origin._y)^2
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_THINGS wxCircleInt : public wxPoint2DInt {
  public:
    inline wxCircleInt(wxInt32 x = 0, wxInt32 y = 0, wxInt32 r = 0) {
        _x = x;
        _y = y;
        _r = r;
    }

    inline wxCircleInt(const wxPoint2DInt& origin, wxInt32 r) {
        _x = origin._x;
        _y = origin._y;
        _r = r;
    }

    inline wxCircleInt(const wxCircleInt& circle) {
        _x = circle._x;
        _y = circle._y;
        _r = circle._r;
    }

    inline bool IsEmpty() const {
        return _r <= 0;
    }

    inline wxInt32 GetX() const {
        return _x;
    }

    inline wxInt32 GetY() const {
        return _y;
    }

    inline wxPoint2DInt GetOrigin() const {
        return wxPoint2DInt(_x, _y);
    }

    inline wxInt32 GetRadius() const {
        return _r;
    }

    // Get a bounding rect
    inline wxRect2DInt GetRect() const {
        return wxRect2DInt(_x - _r, _y - _r, 2 * _r, 2 * _r);
    }

    inline void SetX(wxInt32 x) {
        _x = x;
    }

    inline void SetY(wxInt32 y) {
        _y = y;
    }

    inline void SetOrigin(const wxPoint2DInt& origin) {
        _x = origin._x;
        _y = origin._y;
    }

    inline void SetRadius(wxInt32 r) {
        _r = r;
    }

    inline bool GetPointInCircle(wxInt32 x, wxInt32 y) const {
        if (IsEmpty()) return false;
        return ((x - _x) * (x - _x) + (y - _y) * (y - _y) <= _r * _r);
    }

    inline bool GetPointInCircle(const wxPoint2DInt& pt) const {
        return GetPointInCircle(pt._x, pt._y);
    }

    inline bool Intersects(const wxCircleInt& circle) const {
        return GetDistance(circle) <= _r + circle._r;
    }

    //  int IntersectLine( const wxRay2DDouble &line,
    //                     wxPoint2DInt *pt1=NULL,
    //                     wxPoint2DInt *pt2=NULL ) const;

    // Operators

    inline wxCircleInt operator=(const wxCircleInt& c) {
        _x = c._x;
        _y = c._y;
        _r = c._r;
        return *this;
    }

    inline bool operator==(const wxCircleInt& c) const {
        return (_x == c._x) && (_y == c._y) && (_r == c._r);
    }

    inline bool operator!=(const wxCircleInt& c) const {
        return !(*this == c);
    }

    inline wxCircleInt operator+(const wxPoint2DInt& rel_origin) const {
        return wxCircleInt(_x + rel_origin._x, _y + rel_origin._y, _r);
    }

    inline wxCircleInt operator-(const wxPoint2DInt& rel_origin) const {
        return wxCircleInt(_x - rel_origin._x, _y - rel_origin._y, _r);
    }

    inline wxCircleInt operator*(const wxPoint2DInt& rel_origin) const {
        return wxCircleInt(_x * rel_origin._x, _y * rel_origin._y, _r);
    }

    inline wxCircleInt operator/(const wxPoint2DInt& rel_origin) const {
        return wxCircleInt(_x / rel_origin._x, _y / rel_origin._y, _r);
    }

    inline wxCircleInt& operator+=(const wxPoint2DInt& rel_origin) {
        _x += rel_origin._x;
        _y += rel_origin._y;
        return *this;
    }

    inline wxCircleInt& operator-=(const wxPoint2DInt& rel_origin) {
        _x -= rel_origin._x;
        _y -= rel_origin._y;
        return *this;
    }

    inline wxCircleInt& operator*=(const wxPoint2DInt& rel_origin) {
        _x *= rel_origin._x;
        _y *= rel_origin._y;
        return *this;
    }

    inline wxCircleInt& operator/=(const wxPoint2DInt& rel_origin) {
        _x /= rel_origin._x;
        _y /= rel_origin._y;
        return *this;
    }

    inline wxCircleInt operator+(const wxInt32& rel_radius) const {
        return wxCircleInt(_x, _y, _r + rel_radius);
    }

    inline wxCircleInt operator-(const wxInt32& rel_radius) const {
        return wxCircleInt(_x, _y, _r - rel_radius);
    }

    inline wxCircleInt operator*(const wxInt32& rel_radius) const {
        return wxCircleInt(_x, _y, _r * rel_radius);
    }

    inline wxCircleInt operator/(const wxInt32& rel_radius) const {
        return wxCircleInt(_x, _y, _r / rel_radius);
    }

    inline wxCircleInt& operator+=(const wxInt32& rel_radius) {
        _r += rel_radius;
        return *this;
    }

    inline wxCircleInt& operator-=(const wxInt32& rel_radius) {
        _r -= rel_radius;
        return *this;
    }

    inline wxCircleInt& operator*=(const wxInt32& rel_radius) {
        _r *= rel_radius;
        return *this;
    }

    inline wxCircleInt& operator/=(const wxInt32& rel_radius) {
        _r /= rel_radius;
        return *this;
    }

    wxInt32 _r;
};

//-----------------------------------------------------------------------------
// wxEllipseInt   _r*_r = (x-_origin._x)^2 + (y-_origin._y)^2
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_THINGS wxEllipseInt {
  public:
    inline wxEllipseInt(wxInt32 x = 0, wxInt32 y = 0, wxInt32 r_x = 0, wxInt32 r_y = 0) {
        _origin._x = x;
        _origin._y = y;
        _radius._x = r_x;
        _radius._y = r_y;
    }

    inline wxEllipseInt(const wxPoint2DInt& origin, const wxPoint2DInt radius) {
        _origin = origin;
        _radius = radius;
    }

    inline wxEllipseInt(const wxEllipseInt& ellipse) {
        _origin = ellipse._origin;
        _radius = ellipse._radius;
    }

    inline bool IsEmpty() const {
        return _radius._x <= 0 || _radius._y <= 0;
    }

    inline wxInt32 GetX() const {
        return _origin._x;
    }

    inline wxInt32 GetY() const {
        return _origin._y;
    }

    inline wxPoint2DInt GetOrigin() const {
        return _origin;
    }

    inline wxInt32 GetXRadius() const {
        return _radius._x;
    }

    inline wxInt32 GetYRadius() const {
        return _radius._y;
    }

    inline wxPoint2DInt GetRadius() const {
        return _radius;
    }

    // Get a bounding rect
    inline wxRect2DInt GetRect() const {
        return wxRect2DInt(_origin._x - _radius._x, _origin._y - _radius._y, 2 * _radius._x, 2 * _radius._y);
    }

    inline void SetX(wxInt32 x) {
        _origin._x = x;
    }

    inline void SetY(wxInt32 y) {
        _origin._y = y;
    }

    inline void SetOrigin(const wxPoint2DInt& origin) {
        _origin = origin;
    }

    inline void SetXRadius(wxInt32 r_x) {
        _radius._x = r_x;
    }

    inline void SetYRadius(wxInt32 r_y) {
        _radius._y = r_y;
    }

    inline void SetRadius(const wxPoint2DInt& radius) {
        _radius = radius;
    }

    inline bool GetPointInEllipse(wxInt32 x, wxInt32 y) const {
        if (IsEmpty()) return false;
        return (((x - _origin._x) * (x - _origin._x)) / _radius._x +
                    ((y - _origin._y) * (y - _origin._y)) / _radius._y <=
                1);
    }

    inline bool GetPointInEllipse(const wxPoint2DInt& pt) const {
        return GetPointInEllipse(pt._x, pt._y);
    }

    //  int IntersectLine( const wxRay2DDouble &line,
    //                     wxPoint2DInt *pt1=NULL,
    //                     wxPoint2DInt *pt2=NULL ) const;

    inline bool operator==(const wxEllipseInt& c) const {
        return (_origin == c._origin) && (_radius == c._radius);
    }

    inline bool operator!=(const wxEllipseInt& c) const {
        return !(*this == c);
    }

    inline wxEllipseInt operator+(const wxPoint2DInt& rel_origin) const {
        return wxEllipseInt(_origin + rel_origin, _radius);
    }

    inline wxEllipseInt& operator+=(const wxPoint2DInt& rel_origin) {
        _origin += rel_origin;
        return *this;
    }

    inline wxEllipseInt operator-(const wxPoint2DInt& rel_origin) const {
        return wxEllipseInt(_origin - rel_origin, _radius);
    }

    inline wxEllipseInt& operator-=(const wxPoint2DInt& rel_origin) {
        _origin -= rel_origin;
        return *this;
    }

    wxPoint2DInt _radius;
    wxPoint2DInt _origin;
};

#endif  // __WXIMAGER_GEOMETRY_H__

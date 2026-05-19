///////////////////////////////////////////////////////////////////////////////
// Name:        range.h
// Purpose:     Simple min-max range class and associated selection array class
// Author:      John Labenski
// Created:     12/01/2000
// Copyright:   (c) John Labenski 2004
// Licence:     wxWidgets
///////////////////////////////////////////////////////////////////////////////

#ifndef __WX_RANGE_H__
#define __WX_RANGE_H__

#include "wx/things/thingdef.h"

class WXDLLIMPEXP_THINGS wxRangeInt;

class WXDLLIMPEXP_THINGS wxRangeDouble;

class WXDLLIMPEXP_THINGS wxRangeIntSelection;

class WXDLLIMPEXP_THINGS wxRangeDoubleSelection;

#include "wx/dynarray.h"

WX_DECLARE_OBJARRAY_WITH_DECL(wxRangeInt, wxArrayRangeInt, class WXDLLIMPEXP_THINGS);

WX_DECLARE_OBJARRAY_WITH_DECL(wxRangeDouble, wxArrayRangeDouble, class WXDLLIMPEXP_THINGS);

WX_DECLARE_OBJARRAY_WITH_DECL(wxRangeIntSelection, wxArrayRangeIntSelection, class WXDLLIMPEXP_THINGS);

WX_DECLARE_OBJARRAY_WITH_DECL(wxRangeDoubleSelection, wxArrayRangeDoubleSelection, class WXDLLIMPEXP_THINGS);

// Empty versions of ranges (0, -1)
WXDLLIMPEXP_DATA_THINGS(extern const wxRangeInt) wxEmptyRangeInt;
WXDLLIMPEXP_DATA_THINGS(extern const wxRangeDouble) wxEmptyRangeDouble;

//=============================================================================
// wxRangeInt
//=============================================================================

class WXDLLIMPEXP_THINGS wxRangeInt {
  public:
    inline wxRangeInt(int min_ = 0, int max_ = 0)
        : _min(min_),
          _max(max_) {}

    // Get the width of the range
    inline int GetRange() const {
        return _max - _min + 1;
    }

    // Get/Set the min/max values of the range
    inline int GetMin() const {
        return _min;
    }

    inline int GetMax() const {
        return _max;
    }

    inline void SetMin(int min_) {
        _min = min_;
    }

    inline void SetMax(int max_) {
        _max = max_;
    }

    inline void Set(int min_, int max_) {
        _min = min_, _max = max_;
    }

    // Shift the range by i
    void Shift(int i) {
        _min += i;
        _max += i;
    }

    // Is the range empty, min < max
    inline bool IsEmpty() const {
        return _min > _max;
    }

    // Swap the min and max values
    inline void SwapMinMax() {
        int temp = _min;
        _min = _max;
        _max = temp;
    }

    // returns -1 for i < min, 0 for in range, +1 for i > _max
    inline int Position(int i) const {
        return i < _min ? -1 : (i > _max ? 1 : 0);
    }

    // Is this point or the range within this range
    inline bool Contains(int i) const {
        return (i >= _min) && (i <= _max);
    }

    inline bool Contains(const wxRangeInt& r) const {
        return (r._min >= _min) && (r._max <= _max) && !IsEmpty() && !r.IsEmpty();
    }

    // returns if the range intersects the given range
    inline bool Intersects(const wxRangeInt& r) const {
        return !Intersect(r).IsEmpty();
    }

    // returns the intersection of the range with the other, check IsEmpty()
    inline wxRangeInt Intersect(const wxRangeInt& r) const {
        return wxRangeInt(wxMax(_min, r._min), wxMin(_max, r._max));
    }

    // returns the union of the range with the other, the min and max of the two
    //   regardless of whether they don't overlap
    inline wxRangeInt Union(const wxRangeInt& r) const {
        return (IsEmpty() || r.IsEmpty()) ? wxEmptyRangeInt : wxRangeInt(wxMin(_min, r._min), wxMax(_max, r._max));
    }

    // Is this point inside or touches +/- 1 of the range
    inline bool Touches(int i) const {
        return !IsEmpty() && wxRangeInt(_min - 1, _max + 1).Contains(i);
    }

    // Is the range inside or +/- 1 of this range (eg. is it adjoining?)
    inline bool Touches(const wxRangeInt& r) const {
        return (IsEmpty() || r.IsEmpty()) ? false : r.Intersects(wxRangeInt(_min - 1, _max + 1));
    }

    // combine this single point with the range by expanding the _min/_max to contain it
    //  if only_if_touching then only combine if i is just outside the range by +/-1
    //  returns true if the range has been changed at all, false if not
    bool Combine(int i, bool only_if_touching = false);

    bool Combine(const wxRangeInt& r, bool only_if_touching = false);

    // delete range r from this, return true is anything was done
    //   if r spans this then this and right become wxEmptyRangeInt
    //   else if r is inside of this then this is the left side and right is the right
    //   else if r._min > _min then this is the left side
    //   else if r._min < _min this is the right side
    bool Delete(const wxRangeInt& r, wxRangeInt* right = NULL);

    // operators
    // no copy ctor or assignment operator - the defaults are ok

    // comparison
    inline bool operator==(const wxRangeInt& r) const {
        return (_min == r._min) && (_max == r._max);
    }

    inline bool operator!=(const wxRangeInt& r) const {
        return !(*this == r);
    }

    // Adding ranges unions them to create the largest range
    inline wxRangeInt operator+(const wxRangeInt& r) const {
        return Union(r);
    }

    inline wxRangeInt& operator+=(const wxRangeInt& r) {
        if (r._min < _min) _min = r._min;
        if (r._max > _max) _max = r._max;
        return *this;
    }

    // Subtracting ranges intersects them to get the smallest range
    inline wxRangeInt operator-(const wxRangeInt& r) const {
        return Intersect(r);
    }

    inline wxRangeInt& operator-=(const wxRangeInt& r) {
        if (r._min > _min) _min = r._min;
        if (r._max < _max) _max = r._max;
        return *this;
    }

    // Adding/Subtracting with an int shifts the range
    inline wxRangeInt operator+(const int i) const {
        return wxRangeInt(_min + i, _max + i);
    }

    inline wxRangeInt operator-(const int i) const {
        return wxRangeInt(_min - i, _max - i);
    }

    inline wxRangeInt& operator+=(const int i) {
        Shift(i);
        return *this;
    }

    inline wxRangeInt& operator-=(const int i) {
        Shift(-i);
        return *this;
    }

    int _min, _max;
};

//=============================================================================
// wxRangeIntSelection - ordered 1D array of wxRangeInts, combines to minimze size
//=============================================================================

class WXDLLIMPEXP_THINGS wxRangeIntSelection {
  public:
    wxRangeIntSelection() {}

    wxRangeIntSelection(const wxRangeInt& range) {
        if (!range.IsEmpty()) _ranges.Add(range);
    }

    wxRangeIntSelection(const wxRangeIntSelection& ranges) {
        Copy(ranges);
    }

    // Make a full copy of the source
    void Copy(const wxRangeIntSelection& source) {
        _ranges.Clear();
        WX_APPEND_ARRAY(_ranges, source.GetRangeArray());
    }

    // Get the number of individual ranges
    inline int GetCount() const {
        return _ranges.GetCount();
    }

    // Get total number of items selected in all ranges, ie. sum of all wxRange::GetWidths
    int GetItemCount() const;

    // Get the ranges themselves to iterate though for example
    const wxArrayRangeInt& GetRangeArray() const {
        return _ranges;
    }

    // Get a single range
    const wxRangeInt& GetRange(int index) const;

    inline const wxRangeInt& Item(int index) const {
        return GetRange(index);
    }

    // Get a range of the min range value and max range value
    wxRangeInt GetBoundingRange() const;

    // Clear all the ranges
    void Clear() {
        _ranges.Clear();
    }

    // Is this point or range contained in the selection
    inline bool Contains(int i) const {
        return Index(i) != wxNOT_FOUND;
    }

    inline bool Contains(const wxRangeInt& range) const {
        return Index(range) != wxNOT_FOUND;
    }

    // Get the index of the range that contains this, or wxNOT_FOUND
    int Index(int i) const;

    int Index(const wxRangeInt& range) const;

    // Get the nearest index of a range, index returned contains i or is the one just below
    //   returns -1 if it's below all the selected ones, or no ranges
    //   returns GetCount() if it's above all the selected ones
    int NearestIndex(int i) const;

    // Add the range to the selection, returning if anything was done, false if already selected
    bool SelectRange(const wxRangeInt& range);

    // Remove the range to the selection, returning if anything was done, false if not already selected
    bool DeselectRange(const wxRangeInt& range);

    // Set the min and max bounds of the ranges, returns true if anything was done
    bool BoundRanges(const wxRangeInt& range);

    // operators
    inline const wxRangeInt& operator[](int index) const {
        return GetRange(index);
    }

    wxRangeIntSelection& operator=(const wxRangeIntSelection& other) {
        Copy(other);
        return *this;
    }

  protected:
    wxArrayRangeInt _ranges;
};

//=============================================================================
// wxRangeDouble
//=============================================================================

class WXDLLIMPEXP_THINGS wxRangeDouble {
  public:
    inline wxRangeDouble(wxDouble min_ = 0, wxDouble max_ = 0)
        : _min(min_),
          _max(max_) {}

    // Get the width of the range
    inline wxDouble GetRange() const {
        return _max - _min;
    }

    // Get/Set the min/max values of the range
    inline wxDouble GetMin() const {
        return _min;
    }

    inline wxDouble GetMax() const {
        return _max;
    }

    inline void SetMin(wxDouble min_) {
        _min = min_;
    }

    inline void SetMax(wxDouble max_) {
        _max = max_;
    }

    inline void Set(wxDouble min_, wxDouble max_) {
        _min = min_, _max = max_;
    }

    // Shift the range by i
    void Shift(wxDouble i) {
        _min += i;
        _max += i;
    }

    // Is the range empty, min < max
    inline bool IsEmpty() const {
        return _min > _max;
    }

    // Swap the min and max values
    inline void SwapMinMax() {
        wxDouble temp = _min;
        _min = _max;
        _max = temp;
    }

    // returns -1 for i < min, 0 for in range, +1 for i > _max
    inline int Position(wxDouble i) const {
        return i < _min ? -1 : (i > _max ? 1 : 0);
    }

    // Is this point or the range within this range
    inline bool Contains(wxDouble i) const {
        return (i >= _min) && (i <= _max);
    }

    inline bool Contains(const wxRangeDouble& r) const {
        return (r._min >= _min) && (r._max <= _max) && !IsEmpty() && !r.IsEmpty();
    }

    // returns if the range intersects the given range
    inline bool Intersects(const wxRangeDouble& r) const {
        return !Intersect(r).IsEmpty();
    }

    // returns the intersection of the range with the other, check IsEmpty()
    inline wxRangeDouble Intersect(const wxRangeDouble& r) const {
        return wxRangeDouble(wxMax(_min, r._min), wxMin(_max, r._max));
    }

    // returns the union of the range with the other, the min and max of the two
    //   regardless of whether they don't overlap
    inline wxRangeDouble Union(const wxRangeDouble& r) const {
        return (IsEmpty() || r.IsEmpty()) ? wxEmptyRangeDouble
                                          : wxRangeDouble(wxMin(_min, r._min), wxMax(_max, r._max));
    }

    // no touches for double since what would be a good eps value?

    // combine this single point with the range by expanding the _min/_max to contain it
    //  if only_if_touching then only combine if there is overlap
    //  returns true if the range has been changed at all, false if not
    bool Combine(wxDouble i);

    bool Combine(const wxRangeDouble& r, bool only_if_touching = false);

    // delete range r from this, return true is anything was done
    //   if r spans this then this and right become wxEmptyRangeInt
    //   else if r is inside of this then this is the left side and right is the right
    //   else if r._min > _min then this is the left side
    //   else if r._min < _min this is the right side
    bool Delete(const wxRangeDouble& r, wxRangeDouble* right = NULL);

    // operators
    // no copy ctor or assignment operator - the defaults are ok

    // comparison
    inline bool operator==(const wxRangeDouble& r) const {
        return (_min == r._min) && (_max == r._max);
    }

    inline bool operator!=(const wxRangeDouble& r) const {
        return !(*this == r);
    }

    // Adding ranges unions them to create the largest range
    inline wxRangeDouble operator+(const wxRangeDouble& r) const {
        return Union(r);
    }

    inline wxRangeDouble& operator+=(const wxRangeDouble& r) {
        if (r._min < _min) _min = r._min;
        if (r._max > _max) _max = r._max;
        return *this;
    }

    // Subtracting ranges intersects them to get the smallest range
    inline wxRangeDouble operator-(const wxRangeDouble& r) const {
        return Intersect(r);
    }

    inline wxRangeDouble& operator-=(const wxRangeDouble& r) {
        if (r._min > _min) _min = r._min;
        if (r._max < _max) _max = r._max;
        return *this;
    }

    // Adding/Subtracting with a double shifts the range
    inline wxRangeDouble operator+(const wxDouble i) const {
        return wxRangeDouble(_min + i, _max + i);
    }

    inline wxRangeDouble operator-(const wxDouble i) const {
        return wxRangeDouble(_min - i, _max - i);
    }

    inline wxRangeDouble& operator+=(const wxDouble i) {
        Shift(i);
        return *this;
    }

    inline wxRangeDouble& operator-=(const wxDouble i) {
        Shift(-i);
        return *this;
    }

    wxDouble _min, _max;
};

//=============================================================================
// wxRangeDoubleSelection - ordered 1D array of wxRangeDoubles, combines to minimze size
//=============================================================================

class WXDLLIMPEXP_THINGS wxRangeDoubleSelection {
  public:
    wxRangeDoubleSelection() {}

    wxRangeDoubleSelection(const wxRangeDouble& range) {
        if (!range.IsEmpty()) _ranges.Add(range);
    }

    wxRangeDoubleSelection(const wxRangeDoubleSelection& ranges) {
        Copy(ranges);
    }

    // Make a full copy of the source
    void Copy(const wxRangeDoubleSelection& source) {
        _ranges.Clear();
        WX_APPEND_ARRAY(_ranges, source.GetRangeArray());
    }

    // Get the number of individual ranges
    inline int GetCount() const {
        return _ranges.GetCount();
    }

    // Get the ranges themselves to iterate though for example
    const wxArrayRangeDouble& GetRangeArray() const {
        return _ranges;
    }

    // Get a single range
    const wxRangeDouble& GetRange(int index) const;

    inline const wxRangeDouble& Item(int index) const {
        return GetRange(index);
    }

    // Get a range of the min range value and max range value
    wxRangeDouble GetBoundingRange() const;

    // Clear all the ranges
    void Clear() {
        _ranges.Clear();
    }

    // Is this point or range contained in the selection
    inline bool Contains(wxDouble i) const {
        return Index(i) != wxNOT_FOUND;
    }

    inline bool Contains(const wxRangeDouble& range) const {
        return Index(range) != wxNOT_FOUND;
    }

    // Get the index of the range that contains this, or wxNOT_FOUND
    int Index(wxDouble i) const;

    int Index(const wxRangeDouble& range) const;

    // Get the nearest index of a range, index returned contains i or is the one just below
    //   returns -1 if it's below all the selected ones, or no ranges
    //   returns GetCount() if it's above all the selected ones
    int NearestIndex(wxDouble i) const;

    // Add the range to the selection, returning if anything was done, false if already selected
    bool SelectRange(const wxRangeDouble& range);

    // Remove the range to the selection, returning if anything was done, false if not already selected
    bool DeselectRange(const wxRangeDouble& range);

    // Set the min and max bounds of the ranges, returns true if anything was done
    bool BoundRanges(const wxRangeDouble& range);

    // operators
    inline const wxRangeDouble& operator[](int index) const {
        return GetRange(index);
    }

    wxRangeDoubleSelection& operator=(const wxRangeDoubleSelection& other) {
        Copy(other);
        return *this;
    }

  protected:
    wxArrayRangeDouble _ranges;
};

#endif  // __WX_RANGE_H__

/////////////////////////////////////////////////////////////////////////////
// Name:        range.cpp
// Purpose:     Simple min-max range class and associated selection array class
// Author:      John Labenski
// Created:     12/01/2000
// Copyright:   (c) John Labenski 2004
// Licence:     wxWidgets
/////////////////////////////////////////////////////////////////////////////

// For compilers that support precompilation, includes "wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP

#include "wx/utils.h"

#endif  // WX_PRECOMP

#include <stdio.h>

#include "wx/things/range.h"

const wxRangeInt wxEmptyRangeInt(0, -1);
const wxRangeDouble wxEmptyRangeDouble(0, -1);

#include "wx/arrimpl.cpp"
WX_DEFINE_OBJARRAY(wxArrayRangeInt);
WX_DEFINE_OBJARRAY(wxArrayRangeDouble);
WX_DEFINE_OBJARRAY(wxArrayRangeIntSelection);
WX_DEFINE_OBJARRAY(wxArrayRangeDoubleSelection);

// set this if you want to double check that that ranges are really working
// #define CHECK_RANGES

//=============================================================================
// wxRangeInt
//=============================================================================

bool wxRangeInt::Combine(int i, bool only_if_touching) {
    if (only_if_touching) {
        if (i == _min - 1) {
            _min = i;
            return true;
        } else if (i == _max + 1) {
            _max = i;
            return true;
        }
    } else {
        if (i < _min) {
            _min = i;
            return true;
        } else if (i > _max) {
            _max = i;
            return true;
        }
    }
    return false;
}

bool wxRangeInt::Combine(const wxRangeInt& r, bool only_if_touching) {
    if (only_if_touching) {
        if (Touches(r)) {
            *this += r;
            return true;
        }
    } else if (!IsEmpty() && !r.IsEmpty()) {
        bool added = false;
        if (r._min < _min) {
            _min = r._min;
            added = true;
        }
        if (r._max > _max) {
            _max = r._max;
            added = true;
        }
        return added;
    }
    return false;
}

bool wxRangeInt::Delete(const wxRangeInt& r, wxRangeInt* right) {
    if (!Contains(r)) return false;

    if (right) *right = wxEmptyRangeInt;

    if (r._min <= _min) {
        if (r._max >= _max) {
            *this = wxEmptyRangeInt;
            return true;
        }

        _min = r._max + 1;
        return true;
    }

    if (r._max >= _max) {
        _max = r._min - 1;
        return true;
    }

    if (right) *right = wxRangeInt(r._max + 1, _max);

    _max = r._min - 1;
    return true;
}

//=============================================================================
// wxRangeIntSelection
//=============================================================================
const wxRangeInt& wxRangeIntSelection::GetRange(int index) const {
    wxCHECK_MSG((index >= 0) && (index < int(_ranges.GetCount())), wxEmptyRangeInt, wxT("Invalid index"));
    return _ranges[index];
}

wxRangeInt wxRangeIntSelection::GetBoundingRange() const {
    if (int(_ranges.GetCount()) < 1) return wxEmptyRangeInt;
    return wxRangeInt(_ranges[0]._min, _ranges[_ranges.GetCount() - 1]._max);
}

int wxRangeIntSelection::Index(int i) const {
    int count = _ranges.GetCount();
    if (count < 1) return wxNOT_FOUND;

    if (i < _ranges[0]._min) return wxNOT_FOUND;
    if (i > _ranges[count - 1]._max) return wxNOT_FOUND;

    // Binary search
    int res, tmp, lo = 0, hi = count;

    while (lo < hi) {
        tmp = (lo + hi) / 2;
        res = _ranges[tmp].Position(i);

        if (res == 0)
            return tmp;
        else if (res < 0)
            hi = tmp;
        else  // if ( res > 0 )
            lo = tmp + 1;
    }

    return wxNOT_FOUND;
}

int wxRangeIntSelection::Index(const wxRangeInt& r) const {
    int i, count = _ranges.GetCount();
    for (i = 0; i < count; i++)
        if (_ranges[i].Contains(r)) return i;
    return wxNOT_FOUND;
}

int wxRangeIntSelection::NearestIndex(int i) const {
    int count = _ranges.GetCount();
    if (count < 1) return -1;

    if (i < _ranges[0]._min) return -1;
    if (i > _ranges[count - 1]._max) return count;

    // Binary search
    int res, tmp, lo = 0, hi = count;

    while (lo < hi) {
        tmp = (lo + hi) / 2;
        res = _ranges[tmp].Position(i);

        if (res == 0)
            return tmp;
        else if ((i >= _ranges[tmp]._max) && (i < _ranges[wxMin(tmp + 1, count - 1)]._min))
            return tmp;
        else if (res < 0)
            hi = tmp;
        else  // if ( res > 0 )
            lo = tmp + 1;
    }

    // oops shouldn't get here
    wxCHECK_MSG(0, -1, wxT("Error calculating NearestIndex in wxRangeIntSelection"));
}

int wxRangeIntSelection::GetItemCount() const {
    int i, items = 0, count = _ranges.GetCount();
    for (i = 0; i < count; i++) items += _ranges[i].GetRange();
    return items;
}

bool wxRangeIntSelection::DeselectRange(const wxRangeInt& range) {
    wxCHECK_MSG(!range.IsEmpty(), false, wxT("Invalid Selection Range"));

    bool done = false;
    int i, count = _ranges.GetCount();
    int nearest = count > 0 ? NearestIndex(range._min) : -1;

    if ((nearest < 0) || (nearest == count)) return false;

    wxRangeInt r;
    for (i = nearest; i < int(_ranges.GetCount()); i++) {
        if (range._max < _ranges[i]._min)
            break;
        else if (_ranges[i].Delete(range, &r)) {
            if (_ranges[i].IsEmpty()) {
                _ranges.RemoveAt(i);
                i = (i > 0) ? i - 1 : -1;
            } else if (!r.IsEmpty())
                _ranges.Insert(r, i + 1);

            done = true;
        }
    }

    return done;
}

bool wxRangeIntSelection::SelectRange(const wxRangeInt& range) {
    wxCHECK_MSG(!range.IsEmpty(), false, wxT("Invalid Selection Range"));

    // Try to find a range that includes this one and combine it, else insert it, else append it
    bool done = false;
    int i, count = _ranges.GetCount();
    int nearest = count > 0 ? NearestIndex(range._min) : -1;

    if (nearest < 0) {
        if (!((count > 0) && _ranges[0].Combine(range, true))) _ranges.Insert(range, 0);
        return true;
    } else if (nearest == count) {
        if (!((count > 0) && _ranges[count - 1].Combine(range, true))) _ranges.Add(range);
        return true;
    } else {
        if (_ranges[nearest].Contains(range)) return false;

        for (i = nearest; i < count; i++) {
            if (_ranges[i].Combine(range, true)) {
                done = true;
                break;
            } else if (range._max < _ranges[i]._min) {
                _ranges.Insert(range, i);
                return true;
            }
        }

        count = _ranges.GetCount();
        for (i = wxMax(nearest - 1, 1); i < count; i++) {
            if (range._max + 1 < _ranges[i - 1]._min)
                break;
            else if (_ranges[i - 1].Combine(_ranges[i], true)) {
                _ranges.RemoveAt(i);
                count--;
                i--;
            }
        }
    }

#ifdef CHECK_RANGES
    printf("Selecting ranges %d %d count %d\n", range._min, range._max, _ranges.GetCount());

    for (i = 1; i < int(_ranges.GetCount()); i++) {
        if (_ranges[i - 1].Contains(_ranges[i]))
            printf("Error in Selecting ranges %d %d, %d %d count %d\n", _ranges[i - 1]._min, _ranges[i - 1]._max,
                   _ranges[i]._min, _ranges[i]._max, _ranges.GetCount());
        if (_ranges[i - 1].Touches(_ranges[i]))
            printf("Could have minimzed ranges %d %d, %d %d count %d\n", _ranges[i - 1]._min, _ranges[i - 1]._max,
                   _ranges[i]._min, _ranges[i]._max, _ranges.GetCount());
    }
    fflush(stdout);
#endif  // CHECK_RANGES

    return done;
}

bool wxRangeIntSelection::BoundRanges(const wxRangeInt& range) {
    wxCHECK_MSG(!range.IsEmpty(), false, wxT("Invalid Bounding Range"));
    int i, count = _ranges.GetCount();
    bool done = false;

    for (i = 0; i < count; i++) {
        if (_ranges[i]._min >= range._min) break;

        if (_ranges[i]._max < range._min)  // range is out of bounds
        {
            done = true;
            _ranges.RemoveAt(i);
            count--;
            i--;
        } else {
            done = true;
            _ranges[i]._min = range._min;
            break;
        }
    }

    for (i = _ranges.GetCount() - 1; i >= 0; i--) {
        if (_ranges[i]._max <= range._max) break;

        if (_ranges[i]._min > range._max)  // range is out of bounds
        {
            done = true;
            _ranges.RemoveAt(i);
        } else {
            done = true;
            _ranges[i]._max = range._max;
            break;
        }
    }

    return done;
}

//=============================================================================
// wxRangeDouble
//=============================================================================

bool wxRangeDouble::Combine(double i) {
    if (i < _min) {
        _min = i;
        return true;
    } else if (i > _max) {
        _max = i;
        return true;
    }
    return false;
}

bool wxRangeDouble::Combine(const wxRangeDouble& r, bool only_if_touching) {
    if (only_if_touching) {
        if ((r._min <= _max) && (r._max >= _min))  // Contains(r))
        {
            *this += r;
            return true;
        }
    } else if (!IsEmpty() && !r.IsEmpty()) {
        bool added = false;
        if (r._min < _min) {
            _min = r._min;
            added = true;
        }
        if (r._max > _max) {
            _max = r._max;
            added = true;
        }
        return added;
    }
    return false;
}

bool wxRangeDouble::Delete(const wxRangeDouble& r, wxRangeDouble* right) {
    if (!Contains(r)) return false;

    if (right) *right = wxEmptyRangeDouble;

    if (r._min <= _min) {
        if (r._max >= _max) {
            *this = wxEmptyRangeDouble;
            return true;
        }

        _min = r._max;
        return true;
    }

    if (r._max >= _max) {
        _max = r._min;
        return true;
    }

    if (right) *right = wxRangeDouble(r._max, _max);

    _max = r._min;
    return true;
}

//=============================================================================
// wxRangeDoubleSelection
//=============================================================================
const wxRangeDouble& wxRangeDoubleSelection::GetRange(int index) const {
    wxCHECK_MSG((index >= 0) && (index < int(_ranges.GetCount())), wxEmptyRangeDouble, wxT("Invalid index"));
    return _ranges[index];
}

wxRangeDouble wxRangeDoubleSelection::GetBoundingRange() const {
    if (int(_ranges.GetCount()) < 1) return wxEmptyRangeDouble;
    return wxRangeDouble(_ranges[0]._min, _ranges[_ranges.GetCount() - 1]._max);
}

int wxRangeDoubleSelection::Index(wxDouble i) const {
    int count = _ranges.GetCount();
    if (count < 1) return wxNOT_FOUND;

    if (i < _ranges[0]._min) return wxNOT_FOUND;
    if (i > _ranges[count - 1]._max) return wxNOT_FOUND;

    // Binary search
    int res, tmp, lo = 0, hi = count;

    while (lo < hi) {
        tmp = (lo + hi) / 2;
        res = _ranges[tmp].Position(i);

        if (res == 0)
            return tmp;
        else if (res < 0)
            hi = tmp;
        else  // if ( res > 0 )
            lo = tmp + 1;
    }

    return wxNOT_FOUND;

    /*
        for (int j=0; j<count; j++)
        {
            if (_ranges[j].Contains(i)) return j;
        }
    */
}

int wxRangeDoubleSelection::Index(const wxRangeDouble& r) const {
    int i, count = _ranges.GetCount();
    for (i = 0; i < count; i++)
        if (_ranges[i].Contains(r)) return i;
    return wxNOT_FOUND;
}

int wxRangeDoubleSelection::NearestIndex(wxDouble i) const {
    int count = _ranges.GetCount();
    if (count < 1) return -1;

    if (i < _ranges[0]._min) return -1;
    if (i > _ranges[count - 1]._max) return count;

    // Binary search
    int res, tmp, lo = 0, hi = count;

    while (lo < hi) {
        tmp = (lo + hi) / 2;
        res = _ranges[tmp].Position(i);

        if (res == 0)
            return tmp;
        else if ((i >= _ranges[tmp]._max) && (i < _ranges[wxMin(tmp + 1, count - 1)]._min))
            return tmp;
        else if (res < 0)
            hi = tmp;
        else  // if ( res > 0 )
            lo = tmp + 1;
    }

    // oops shouldn't get here
    wxCHECK_MSG(0, -1, wxT("Error calculating NearestIndex in wxRangeDoubleSelection"));
}

bool wxRangeDoubleSelection::SelectRange(const wxRangeDouble& range) {
    wxCHECK_MSG(!range.IsEmpty(), false, wxT("Invalid Selection Range"));

    // Try to find a range that includes this one and combine it, else insert it, else append it
    bool done = false;
    int i, count = _ranges.GetCount();
    int nearest = count > 0 ? NearestIndex(range._min) : -1;

    if (nearest < 0) {
        if (!((count > 0) && _ranges[0].Combine(range, true))) _ranges.Insert(range, 0);
        return true;
    } else if (nearest == count) {
        if (!((count > 0) && _ranges[count - 1].Combine(range, true))) _ranges.Add(range);
        return true;
    } else {
        if (_ranges[nearest].Contains(range)) return false;

        for (i = nearest; i < count; i++) {
            if (_ranges[i].Combine(range, true)) {
                done = true;
                break;
            } else if (range._max < _ranges[i]._min) {
                _ranges.Insert(range, i);
                return true;
            }
        }
        for (i = wxMax(nearest - 1, 1); i < int(_ranges.GetCount()); i++) {
            if (range._max + 1 < _ranges[i - 1]._min)
                break;
            else if (_ranges[i - 1].Combine(_ranges[i], true)) {
                _ranges.RemoveAt(i);
                i--;
            }
        }
    }

#ifdef CHECK_RANGES
    printf("Selecting ranges %g %g count %d\n", range._min, range._max, _ranges.GetCount());

    for (i = 1; i < int(_ranges.GetCount()); i++) {
        if (_ranges[i - 1].Contains(_ranges[i]))
            printf("Error in Selecting ranges %g %g, %g %g count %d\n", _ranges[i - 1]._min, _ranges[i - 1]._max,
                   _ranges[i]._min, _ranges[i]._max, _ranges.GetCount());
        // if (_ranges[i-1].Touches(_ranges[i]))
        //    printf("Could have minimzed ranges %g %g, %g %g count %d\n", _ranges[i-1]._min, _ranges[i-1]._max,
        //    _ranges[i]._min, _ranges[i]._max, _ranges.GetCount());
    }
    fflush(stdout);
#endif  // CHECK_RANGES

    return done;
}

bool wxRangeDoubleSelection::DeselectRange(const wxRangeDouble& range) {
    wxCHECK_MSG(!range.IsEmpty(), false, wxT("Invalid Selection Range"));

    bool done = false;
    int i, count = _ranges.GetCount();
    int nearest = count > 0 ? NearestIndex(range._min) : -1;

    if ((nearest < 0) || (nearest == count)) return false;

    wxRangeDouble r;
    for (i = nearest; i < int(_ranges.GetCount()); i++) {
        if (range._max < _ranges[i]._min)
            break;
        else if (_ranges[i].Delete(range, &r)) {
            if (_ranges[i].IsEmpty()) {
                _ranges.RemoveAt(i);
                i = (i > 0) ? i - 1 : -1;
            } else if (!r.IsEmpty())
                _ranges.Insert(r, i + 1);

            done = true;
        }
    }

    return done;
}

bool wxRangeDoubleSelection::BoundRanges(const wxRangeDouble& range) {
    wxCHECK_MSG(!range.IsEmpty(), false, wxT("Invalid Bounding Range"));
    int i, count = _ranges.GetCount();
    bool done = false;

    for (i = 0; i < count; i++) {
        if (_ranges[i]._min >= range._min) break;

        if (_ranges[i]._max < range._min)  // range is out of bounds
        {
            done = true;
            _ranges.RemoveAt(i);
            count--;
            i--;
        } else {
            done = true;
            _ranges[i]._min = range._min;
            break;
        }
    }

    for (i = _ranges.GetCount() - 1; i >= 0; i--) {
        if (_ranges[i]._max <= range._max) break;

        if (_ranges[i]._min > range._max)  // range is out of bounds
        {
            done = true;
            _ranges.RemoveAt(i);
        } else {
            done = true;
            _ranges[i]._max = range._max;
            break;
        }
    }

    return done;
}

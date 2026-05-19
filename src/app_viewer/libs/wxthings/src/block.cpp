/////////////////////////////////////////////////////////////////////////////
// Name:        block.cpp
// Purpose:     Rectangular selection storage classes for ints and doubles
// Author:      John Labenski
// Created:     07/01/02
// Copyright:   (c) John Labenski 2004
// Licence:     wxWidgets
/////////////////////////////////////////////////////////////////////////////

// For compilers that support precompilation, includes "wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
// #include "wx/object.h"
#endif  // WX_PRECOMP

#include "wx/things/block.h"

// use this to check to see if there is any overlap after minimizing
// #define CHECK_BLOCK_OVERLAP 1

#define PRINT_BLOCK(msg, b)                                                                                      \
    {                                                                                                            \
        wxPrintf(wxT("Block '%s' %lg %lg %lg %lg\n"), msg, (double)(b)._x1, (double)(b)._y1, (double)(b)._x2, \
                 (double)(b)._y2);                                                                              \
    }

wxBlockInt const wxEmptyBlockInt(0, 0, -1, -1);
wxBlockDouble const wxEmptyBlockDouble(0, 0, -1, -1);

#include "wx/arrimpl.cpp"
WX_DEFINE_OBJARRAY(wxArrayBlockInt);
WX_DEFINE_OBJARRAY(wxArrayBlockDouble);
WX_DEFINE_OBJARRAY(wxArrayBlockIntSelection);
WX_DEFINE_OBJARRAY(wxArrayBlockDoubleSelection);

// ----------------------------------------------------------------------------
// Sorting functions for wxBlockInt
// ----------------------------------------------------------------------------

static int wxCMPFUNC_CONV wxblockint_sort_topleft_bottomright(wxBlockInt** a, wxBlockInt** b) {
    int y = ((*a)->_y1 - (*b)->_y1);

    if (y < 0) return -1;
    if (y == 0) return ((*a)->_x1 - (*b)->_x1);
    return 1;
}

static int wxCMPFUNC_CONV wxblockint_sort_topright_bottomleft(wxBlockInt** a, wxBlockInt** b) {
    int y = ((*a)->_y1 - (*b)->_y1);

    if (y < 0) return -1;
    if (y == 0) return ((*a)->_x2 - (*b)->_x2);
    return 1;
}

static int wxCMPFUNC_CONV wxblockint_sort_bottomleft_topright(wxBlockInt** a, wxBlockInt** b) {
    int y = ((*a)->_y2 - (*b)->_y2);

    if (y > 0) return -1;
    if (y == 0) return ((*a)->_x1 - (*b)->_x1);
    return 1;
}

static int wxCMPFUNC_CONV wxblockint_sort_bottomright_topleft(wxBlockInt** a, wxBlockInt** b) {
    int y = ((*a)->_y2 - (*b)->_y2);

    if (y > 0) return -1;
    if (y == 0) return ((*a)->_x2 - (*b)->_x2);
    return 1;
}

static int wxCMPFUNC_CONV wxblockint_sort_largest_to_smallest(wxBlockInt** a, wxBlockInt** b) {
    return (*a)->IsLarger(**b);
}

static int wxCMPFUNC_CONV wxblockint_sort_smallest_to_largest(wxBlockInt** a, wxBlockInt** b) {
    return -(*a)->IsLarger(**b);
}

void wxArrayBlockIntSort(wxArrayBlockInt& blocks, wxBlockSort_Type type) {
    switch (type) {
        case wxBLOCKSORT_TOPLEFT_BOTTOMRIGHT:
            blocks.Sort(wxblockint_sort_topleft_bottomright);
            break;
        case wxBLOCKSORT_TOPRIGHT_BOTTOMLEFT:
            blocks.Sort(wxblockint_sort_topright_bottomleft);
            break;
        case wxBLOCKSORT_BOTTOMLEFT_TOPRIGHT:
            blocks.Sort(wxblockint_sort_bottomleft_topright);
            break;
        case wxBLOCKSORT_BOTTOMRIGHT_TOPLEFT:
            blocks.Sort(wxblockint_sort_bottomright_topleft);
            break;
        case wxBLOCKSORT_SMALLEST_TO_LARGEST:
            blocks.Sort(wxblockint_sort_smallest_to_largest);
            break;
        case wxBLOCKSORT_LARGEST_TO_SMALLEST:
            blocks.Sort(wxblockint_sort_largest_to_smallest);
            break;
        default:
            wxFAIL_MSG(wxT("unknown block sort type"));
    }
}

// ----------------------------------------------------------------------------
// Sorting functions for wxBlockDouble
// ----------------------------------------------------------------------------

static int wxCMPFUNC_CONV wxblockdouble_sort_topleft_bottomright(wxBlockDouble** a, wxBlockDouble** b) {
    wxDouble y = ((*a)->_y1 - (*b)->_y1);

    if (y < 0) return -1;
    if (y == 0) return int((*a)->_x1 - (*b)->_x1);
    return 1;
}

static int wxCMPFUNC_CONV wxblockdouble_sort_topright_bottomleft(wxBlockDouble** a, wxBlockDouble** b) {
    wxDouble y = ((*a)->_y1 - (*b)->_y1);

    if (y < 0) return -1;
    if (y == 0) return int((*a)->_x2 - (*b)->_x2);
    return 1;
}

static int wxCMPFUNC_CONV wxblockdouble_sort_bottomleft_topright(wxBlockDouble** a, wxBlockDouble** b) {
    wxDouble y = ((*a)->_y2 - (*b)->_y2);

    if (y > 0) return -1;
    if (y == 0) return int((*a)->_x1 - (*b)->_x1);
    return 1;
}

static int wxCMPFUNC_CONV wxblockdouble_sort_bottomright_topleft(wxBlockDouble** a, wxBlockDouble** b) {
    wxDouble y = ((*a)->_y2 - (*b)->_y2);

    if (y > 0) return -1;
    if (y == 0) return int((*a)->_x2 - (*b)->_x2);
    return 1;
}

static int wxCMPFUNC_CONV wxblockdouble_sort_largest_to_smallest(wxBlockDouble** a, wxBlockDouble** b) {
    return (*a)->IsLarger(**b);
}

static int wxCMPFUNC_CONV wxblockdouble_sort_smallest_to_largest(wxBlockDouble** a, wxBlockDouble** b) {
    return -(*a)->IsLarger(**b);
}

void wxArrayBlockDoubleSort(wxArrayBlockDouble& blocks, wxBlockSort_Type type) {
    switch (type) {
        case wxBLOCKSORT_TOPLEFT_BOTTOMRIGHT:
            blocks.Sort(wxblockdouble_sort_topleft_bottomright);
            break;
        case wxBLOCKSORT_TOPRIGHT_BOTTOMLEFT:
            blocks.Sort(wxblockdouble_sort_topright_bottomleft);
            break;
        case wxBLOCKSORT_BOTTOMLEFT_TOPRIGHT:
            blocks.Sort(wxblockdouble_sort_bottomleft_topright);
            break;
        case wxBLOCKSORT_BOTTOMRIGHT_TOPLEFT:
            blocks.Sort(wxblockdouble_sort_bottomright_topleft);
            break;
        case wxBLOCKSORT_SMALLEST_TO_LARGEST:
            blocks.Sort(wxblockdouble_sort_smallest_to_largest);
            break;
        case wxBLOCKSORT_LARGEST_TO_SMALLEST:
            blocks.Sort(wxblockdouble_sort_largest_to_smallest);
            break;
        default:
            wxFAIL_MSG(wxT("unknown block sort type"));
    }
}

//=============================================================================
// wxBlockInt
//=============================================================================

int wxBlockInt::IsLarger(const wxBlockInt& b) const {
    wxInt32 width = _x2 - _x1 + 1, height = _y2 - _y1 + 1, b_width = b._x2 - b._x1 + 1,
            b_height = b._y2 - b._y1 + 1;

    if ((width <= 0) || (height <= 0)) return (b_width > 0) && (b_height > 0) ? -1 : 0;
    if ((b_width <= 0) || (b_height <= 0)) return (width > 0) && (height > 0) ? 1 : 0;

    wxDouble w_bw = wxDouble(width) / b_width, bh_h = wxDouble(b_height) / height;

    return (w_bw == bh_h) ? 0 : ((w_bw > bh_h) ? 1 : -1);
}

bool wxBlockInt::Touches(const wxBlockInt& b) const  // see Intersects
{
    return Intersects(wxBlockInt(b._x1 - 1, b._y1 - 1, b._x2 + 1, b._y2 + 1));
}

bool wxBlockInt::Combine(const wxBlockInt& b) {
    if (!Touches(b)) return false;
    if (Contains(b)) return true;
    if (b.Contains(*this)) {
        *this = b;
        return true;
    }

    wxBlockInt unionBlock;
    Union(*this, b, &unionBlock);

    if (unionBlock.IsEmpty()) return false;

    // at least one of the two blocks has to be at each corner of the union
    if (((unionBlock.GetLeftTop() == GetLeftTop()) || (unionBlock.GetLeftTop() == b.GetLeftTop())) &&
        ((unionBlock.GetRightTop() == GetRightTop()) || (unionBlock.GetRightTop() == b.GetRightTop())) &&
        ((unionBlock.GetLeftBottom() == GetLeftBottom()) || (unionBlock.GetLeftBottom() == b.GetLeftBottom())) &&
        ((unionBlock.GetRightBottom() == GetRightBottom()) || (unionBlock.GetRightBottom() == b.GetRightBottom()))) {
        *this = unionBlock;
        return true;
    }

    return false;
}

bool wxBlockInt::Combine(const wxBlockInt& block, wxBlockInt& top, wxBlockInt& bottom, wxBlockInt& left,
                         wxBlockInt& right) const {
    top = bottom = left = right = wxEmptyBlockInt;

    wxBlockInt iBlock;
    Intersect(*this, block, &iBlock);

    if (iBlock.IsEmpty()) return false;  // nothing to combine
    if (iBlock == *this) return true;    // can combine all of this, no leftover

    bool combined = false;

    if (block._y1 < _y1) {
        top = wxBlockInt(block._x1, block._y1, block._x2, _y1 - 1);
        combined = true;
    }
    if (block._y2 > _y2) {
        bottom = wxBlockInt(block._x1, _y2 + 1, block._x2, block._y2);
        combined = true;
    }
    if (block._x1 < _x1) {
        left = wxBlockInt(block._x1, iBlock._y1, _x1 - 1, iBlock._y2);
        combined = true;
    }
    if (block._x2 > _x2) {
        right = wxBlockInt(_x2 + 1, iBlock._y1, block._x2, iBlock._y2);
        combined = true;
    }

    return combined;
}

bool wxBlockInt::Delete(const wxBlockInt& block, wxBlockInt& top, wxBlockInt& bottom, wxBlockInt& left,
                        wxBlockInt& right) const {
    top = bottom = left = right = wxEmptyBlockInt;

    wxBlockInt iBlock;
    Intersect(*this, block, &iBlock);

    if (iBlock.IsEmpty()) return false;  // nothing to delete
    if (iBlock == *this) return true;    // can delete all of this, no leftover

    bool deleted = false;

    if (_y1 < iBlock._y1) {
        top = wxBlockInt(_x1, _y1, _x2, iBlock._y1 - 1);
        deleted = true;
    }
    if (GetBottom() > iBlock.GetBottom()) {
        bottom = wxBlockInt(_x1, iBlock._y2 + 1, _x2, _y2);
        deleted = true;
    }
    if (_x1 < iBlock._x1) {
        left = wxBlockInt(_x1, iBlock._y1, iBlock._x1 - 1, iBlock._y2);
        deleted = true;
    }
    if (GetRight() > iBlock.GetRight()) {
        right = wxBlockInt(iBlock._x2 + 1, iBlock._y1, _x2, iBlock._y2);
        deleted = true;
    }

    return deleted;
}

//=============================================================================
// wxBlockDouble
//=============================================================================

int wxBlockDouble::IsLarger(const wxBlockDouble& b) const {
    wxDouble width = _x2 - _x1, height = _y2 - _y1, b_width = b._x2 - b._x1, b_height = b._y2 - b._y1;

    if ((width <= 0) || (height <= 0)) return (b_width > 0) && (b_height > 0) ? -1 : 0;
    if ((b_width <= 0) || (b_height <= 0)) return (width > 0) && (height > 0) ? 1 : 0;

    wxDouble w_bw = width / b_width, bh_h = b_height / height;
    return (w_bw == bh_h) ? 0 : ((w_bw > bh_h) ? 1 : -1);
}

bool wxBlockDouble::Touches(const wxBlockDouble& b) const  // see Intersects
{
    if (((wxMax(_x1, b._x1)) <= (wxMin(_x2, b._x2))) && ((wxMax(_y1, b._y1)) <= (wxMin(_y2, b._y2))))
        return true;

    return false;
}

bool wxBlockDouble::Combine(const wxBlockDouble& b) {
    if (!Touches(b)) return false;
    if (Contains(b)) return true;
    if (b.Contains(*this)) {
        *this = b;
        return true;
    }

    wxBlockDouble unionBlock;
    Union(*this, b, &unionBlock);

    if (unionBlock.IsEmpty()) return false;

    // at least one of the two blocks has to be at each corner of the union
    if (((unionBlock.GetLeftTop() == GetLeftTop()) || (unionBlock.GetLeftTop() == b.GetLeftTop())) &&
        ((unionBlock.GetRightTop() == GetRightTop()) || (unionBlock.GetRightTop() == b.GetRightTop())) &&
        ((unionBlock.GetLeftBottom() == GetLeftBottom()) || (unionBlock.GetLeftBottom() == b.GetLeftBottom())) &&
        ((unionBlock.GetRightBottom() == GetRightBottom()) || (unionBlock.GetRightBottom() == b.GetRightBottom()))) {
        *this = unionBlock;
        return true;
    }

    return false;
}

bool wxBlockDouble::Combine(const wxBlockDouble& block, wxBlockDouble& top, wxBlockDouble& bottom, wxBlockDouble& left,
                            wxBlockDouble& right) const {
    top = bottom = left = right = wxEmptyBlockDouble;

    wxBlockDouble iBlock;
    Intersect(*this, block, &iBlock);

    if (iBlock.IsEmpty()) return false;  // nothing to combine
    if (iBlock == *this) return true;    // can combine all of this, no leftover

    bool combined = false;

    if (block._y1 < _y1) {
        top = wxBlockDouble(block._x1, block._y1, block._x2, _y1);
        combined = true;
    }
    if (block._y2 > _y2) {
        bottom = wxBlockDouble(block._x1, _y2, block._x2, block._y2);
        combined = true;
    }
    if (block._x1 < _x1) {
        left = wxBlockDouble(block._x1, iBlock._y1, _x1, iBlock._y2);
        combined = true;
    }
    if (block._x2 > _x2) {
        right = wxBlockDouble(_x2, iBlock._y1, block._x2, iBlock._y2);
        combined = true;
    }

    return combined;
}

bool wxBlockDouble::Delete(const wxBlockDouble& block, wxBlockDouble& top, wxBlockDouble& bottom, wxBlockDouble& left,
                           wxBlockDouble& right) const {
    top = bottom = left = right = wxEmptyBlockDouble;

    wxBlockDouble iBlock;
    Intersect(*this, block, &iBlock);

    if (iBlock.IsEmpty()) return false;  // nothing to delete
    if (iBlock == *this) return true;    // can delete all of this, no leftover

    bool deleted = false;

    if (_y1 < iBlock._y1) {
        top = wxBlockDouble(_x1, _y1, _x2, iBlock._y1);
        deleted = true;
    }
    if (_y2 > iBlock._y2) {
        bottom = wxBlockDouble(_x1, iBlock._y2, _x2, _y2);
        deleted = true;
    }
    if (_x1 < iBlock._x1) {
        left = wxBlockDouble(_x1, iBlock._y1, iBlock._x1, iBlock._y2);
        deleted = true;
    }
    if (_x2 > iBlock._x2) {
        right = wxBlockDouble(iBlock._x2, iBlock._y1, _x2, iBlock._y2);
        deleted = true;
    }

    return deleted;
}

//=============================================================================
// wxBlockIntSelection
//=============================================================================
wxBlockInt wxBlockIntSelection::GetBlock(int index) const {
    wxCHECK_MSG((index >= 0) && (index < int(_blocks.GetCount())), wxEmptyBlockInt, wxT("Invalid index"));
    return _blocks[index];
}

#ifdef USE_wxRANGE
wxArrayRangeInt wxBlockIntSelection::GetBlockCol(int col) const {
    wxArrayRangeInt ranges;
    int n, count = _blocks.GetCount();
    for (n = 0; n < count; n++) {
        if ((col >= _blocks[n]._x1) && (col <= _blocks[n]._x2)) {
            wxRangeInt range(_blocks[n]._y1, _blocks[n]._y2);
            ranges.Add(range);
        }
    }
    return ranges;
}

wxArrayRangeInt wxBlockIntSelection::GetBlockRow(int row) const {
    wxArrayRangeInt ranges;
    int n, count = _blocks.GetCount();
    for (n = 0; n < count; n++) {
        if ((row >= _blocks[n]._y1) && (row <= _blocks[n]._y2))
            ranges.Add(wxRangeInt(_blocks[n]._x1, _blocks[n]._x2));
    }
    return ranges;
}
#endif  // USE_wxRANGE

wxBlockInt wxBlockIntSelection::GetBoundingBlock() const {
    int n, count = _blocks.GetCount();
    if (count == 0) return wxEmptyBlockInt;
    wxBlockInt bound = _blocks[0];
    for (n = 1; n < count; n++) bound.Union(_blocks[n]);
    return bound;
}

int wxBlockIntSelection::Index(int x, int y) const {
    int n, count = _blocks.GetCount();
    for (n = 0; n < count; n++) {
        if (_blocks[n].Contains(x, y)) return n;
    }
    return wxNOT_FOUND;
}

int wxBlockIntSelection::Index(const wxBlockInt& b) const {
    int n, count = _blocks.GetCount();
    for (n = 0; n < count; n++) {
        if (_blocks[n].Intersects(b)) return n;
    }
    return wxNOT_FOUND;
}

void wxBlockIntSelection::Sort(wxBlockSort_Type type) {
    _sort = type;
    wxArrayBlockIntSort(_blocks, type);
}

bool wxBlockIntSelection::DeselectBlock(const wxBlockInt& block, bool combineNow) {
    wxCHECK_MSG(!block.IsEmpty(), false, wxT("Invalid block"));

    bool done = false;

    wxBlockInt top, bottom, left, right;
    for (int n = 0; n < int(_blocks.GetCount()); n++) {
        if (_blocks[n].Delete(block, top, bottom, left, right)) {
            done = true;
            _blocks.RemoveAt(n);
            n = (n > 0) ? n - 1 : -1;

            if (!top.IsEmpty()) _blocks.Add(top);
            if (!bottom.IsEmpty()) _blocks.Add(bottom);
            if (!left.IsEmpty()) _blocks.Add(left);
            if (!right.IsEmpty()) _blocks.Add(right);
        }
    }

    if (combineNow) Minimize();

    return done;
}

bool wxBlockIntSelection::SelectBlock(const wxBlockInt& block, bool combineNow, wxArrayBlockInt* addedBlocks) {
    wxCHECK_MSG(!block.IsEmpty(), false, wxT("Invalid block"));

    // TestBlocks();

    wxArrayBlockInt extraBlocks;
    wxArrayBlockInt* extra = &extraBlocks;

    if (addedBlocks != NULL) {
        addedBlocks->Clear();
        extra = addedBlocks;
    }

    extra->Add(block);

    int n, count = _blocks.GetCount();
    wxBlockInt top, bottom, left, right;

    for (n = 0; n < count; n++) {
        for (int k = 0; k < int(extra->GetCount()); k++) {
            if (_blocks[n].Combine(extra->Item(k), top, bottom, left, right)) {
                extra->RemoveAt(k);
                if (!top.IsEmpty()) extra->Add(top);
                if (!bottom.IsEmpty()) extra->Add(bottom);
                if (!left.IsEmpty()) extra->Add(left);
                if (!right.IsEmpty()) extra->Add(right);
                // DoMinimize( *extra );
                n = -1;
                break;
            }
        }
    }

    if (extra->GetCount() > 0u) {
        WX_APPEND_ARRAY(_blocks, *extra);
        if (combineNow) Minimize();

        return true;
    }

    return false;
}

bool wxBlockIntSelection::Minimize() {
    bool ret = DoMinimize(_blocks);
    Sort(_sort);
    return ret;
}

bool wxBlockIntSelection::DoMinimize(wxArrayBlockInt& blocks) {
    int n;
    for (n = 0; n < 1000; n++)  // should probably just take a few
    {
        if (!DoDoMinimize(blocks)) break;
    }

#ifdef CHECK_BLOCK_OVERLAP
    for (size_t a = 0; a < blocks.GetCount(); a++) {
        for (size_t b = a + 1; b < blocks.GetCount(); b++) {
            if (blocks[a].Intersects(blocks[b])) {
                printf("Intersecting blocks in wxBlockIntSelection::DoMinimize");
                fflush(stdout);
                wxBell();
            }
        }
    }
#endif

    return n != 0;
}

bool wxBlockIntSelection::DoDoMinimize(wxArrayBlockInt& blocks) {
    //    wxBlockInt top, bottom, left, right;
    bool done = false;
    for (int i = 0; i < int(blocks.GetCount()) - 1; i++) {
        for (int j = i + 1; j < int(blocks.GetCount()); j++) {
            if (blocks[i].Combine(blocks[j])) {
                blocks.RemoveAt(j);
                j--;
                done = true;
                // return true;
            }
            /*
                        else if (blocks[i].Combine(blocks[j], top, bottom, left, right))
                        {
                            printf("INTERSECTION!?---------------------------\n"); fflush(stdout);
                            blocks.RemoveAt(j);
                            if (!top.IsEmpty())    blocks.Add(top);
                            if (!bottom.IsEmpty()) blocks.Add(bottom);
                            if (!left.IsEmpty())   blocks.Add(left);
                            if (!right.IsEmpty())  blocks.Add(right);
                            return true;
                        }
            */
        }
    }
    return done;
}

//=============================================================================
// wxBlockDoubleSelection
//=============================================================================
wxBlockDouble wxBlockDoubleSelection::GetBlock(int index) const {
    wxCHECK_MSG((index >= 0) && (index < int(_blocks.GetCount())), wxEmptyBlockDouble, wxT("Invalid index"));
    return _blocks[index];
}

#ifdef USE_wxRANGE
wxArrayRangeDouble wxBlockDoubleSelection::GetBlockCol(wxDouble col) const {
    wxArrayRangeDouble ranges;
    int n, count = _blocks.GetCount();
    for (n = 0; n < count; n++) {
        if ((col >= _blocks[n]._x1) && (col <= _blocks[n]._x2)) {
            wxRangeDouble range(_blocks[n]._y1, _blocks[n]._y2);
            ranges.Add(range);
        }
    }
    return ranges;
}

wxArrayRangeDouble wxBlockDoubleSelection::GetBlockRow(wxDouble row) const {
    wxArrayRangeDouble ranges;
    int n, count = _blocks.GetCount();
    for (n = 0; n < count; n++) {
        if ((row >= _blocks[n]._y1) && (row <= _blocks[n]._y2))
            ranges.Add(wxRangeDouble(_blocks[n]._x1, _blocks[n]._x2));
    }
    return ranges;
}
#endif  // USE_wxRANGE

wxBlockDouble wxBlockDoubleSelection::GetBoundingBlock() const {
    int n, count = _blocks.GetCount();
    if (count == 0) return wxEmptyBlockDouble;
    wxBlockDouble bound = _blocks[0];
    for (n = 1; n < count; n++) bound.Union(_blocks[n]);
    return bound;
}

int wxBlockDoubleSelection::Index(wxDouble x, wxDouble y) const {
    int n, count = _blocks.GetCount();
    for (n = 0; n < count; n++) {
        if ((x >= _blocks[n]._x1) && (y >= _blocks[n]._y1) && (x <= _blocks[n]._x2) && (y <= _blocks[n]._y2))
            return true;
    }
    return wxNOT_FOUND;
}

int wxBlockDoubleSelection::Index(const wxBlockDouble& b) const {
    int n, count = _blocks.GetCount();
    for (n = 0; n < count; n++) {
        if (_blocks[n].Intersects(b)) return n;
    }
    return wxNOT_FOUND;
}

void wxBlockDoubleSelection::Sort(wxBlockSort_Type type) {
    _sort = type;
    wxArrayBlockDoubleSort(_blocks, type);
}

bool wxBlockDoubleSelection::DeselectBlock(const wxBlockDouble& block, bool combineNow) {
    // wxCHECK_MSG(!block.IsEmpty(), false, wxT("Invalid block") );

    bool done = false;

    wxBlockDouble top, bottom, left, right;
    for (int n = 0; n < int(_blocks.GetCount()); n++) {
        if (_blocks[n].Delete(block, top, bottom, left, right)) {
            done = true;
            _blocks.RemoveAt(n);
            n = (n > 0) ? n - 1 : -1;

            if (!top.IsEmpty()) _blocks.Add(top);
            if (!bottom.IsEmpty()) _blocks.Add(bottom);
            if (!left.IsEmpty()) _blocks.Add(left);
            if (!right.IsEmpty()) _blocks.Add(right);
        }
    }

    if (combineNow) Minimize();

    return done;
}

bool wxBlockDoubleSelection::SelectBlock(const wxBlockDouble& block, bool combineNow) {
    // It's valid to select a block with a width and height 0 since that means that point
    // wxCHECK_MSG(!block.IsEmpty(), false, wxT("Invalid block") );

    wxArrayBlockDouble extra;
    extra.Add(block);
    wxBlockDouble top, bottom, left, right;

    for (int n = 0; n < int(_blocks.GetCount()); n++) {
        for (int k = 0; k < int(extra.GetCount()); k++) {
            bool done = false;

            // Doubles are different than ints - roundoff error problems
            // always use the bigger block to soak up the smaller blocks
            // this reduces problems with tiny roundoff error produced blocks
            if (_blocks[n].Intersects(extra[k])) {
                if (_blocks[n].Contains(extra[k])) {
                    extra.RemoveAt(k);
                    k--;
                    continue;
                } else if (extra[k].Contains(_blocks[n])) {
                    _blocks.RemoveAt(n);
                    n = -1;
                    break;
                } else if (_blocks[n].IsLarger(extra[k]) > 0) {
                    done = _blocks[n].Combine(extra[k], top, bottom, left, right);
                    if (done) {
                        extra.RemoveAt(k);
                        k--;
                    }
                } else {
                    done = extra[k].Combine(_blocks[n], top, bottom, left, right);
                    if (done) {
                        _blocks.RemoveAt(n);
                        n = -1;
                    }
                }
            }

            if (done) {
                if (!top.IsEmpty()) extra.Add(top);
                if (!bottom.IsEmpty()) extra.Add(bottom);
                if (!left.IsEmpty()) extra.Add(left);
                if (!right.IsEmpty()) extra.Add(right);
                // DoMinimize( extra );
                if (n == -1) break;
            }
        }
    }

    if (extra.GetCount() > 0u) {
        WX_APPEND_ARRAY(_blocks, extra);
        if (combineNow) Minimize();

        return true;
    }

    return false;
}

bool wxBlockDoubleSelection::Minimize() {
    bool ret = DoMinimize(_blocks);
    Sort(_sort);
    return ret;
}

bool wxBlockDoubleSelection::DoMinimize(wxArrayBlockDouble& blocks) {
    int n;
    for (n = 0; n < 1000; n++)  // should probably just take < 10 at most
    {
        if (!DoDoMinimize(blocks)) break;
    }

#ifdef CHECK_BLOCK_OVERLAP
    for (size_t a = 0; a < blocks.GetCount(); a++) {
        printf("Checking wxBlockDoubleSelection::DoMinimize %d =", a);
        PRINT_BLOCK("", blocks[a])
        for (size_t b = a + 1; b < blocks.GetCount(); b++) {
            if (blocks[a].Intersects(blocks[b])) {
                printf("Intersecting blocks in wxBlockDoubleSelection::DoMinimize\n");
                fflush(stdout);
                PRINT_BLOCK("", blocks[a])
                PRINT_BLOCK("", blocks[b])
                wxBell();
            }
        }
    }
#endif

    return n != 0;
}

bool wxBlockDoubleSelection::DoDoMinimize(wxArrayBlockDouble& blocks) {
    // wxBlockDouble top, bottom, left, right;
    bool done = false;

    for (int i = 0; i < int(blocks.GetCount()) - 1; i++) {
        for (int j = i + 1; j < int(blocks.GetCount()); j++) {
            if (blocks[i].Combine(blocks[j])) {
                blocks.RemoveAt(j);
                done = true;
                j--;
            }
            /*
                        else if (blocks[i].Combine(blocks[j], top, bottom, left, right))
                        {
                            blocks.RemoveAt(j);
                            if (!top.IsEmpty())    blocks.Add(top);
                            if (!bottom.IsEmpty()) blocks.Add(bottom);
                            if (!left.IsEmpty())   blocks.Add(left);
                            if (!right.IsEmpty())  blocks.Add(right);
                            return true;
                        }
            */
        }
    }
    return done;
}

//=============================================================================
// wxBlockIntSelectionIterator - iterates through a wxBlockIntSelection
//=============================================================================

wxBlockIntSelectionIterator::wxBlockIntSelectionIterator(const wxBlockIntSelection& sel, wxBLOCKINT_SELITER_Type type) {
    _type = type;
    WX_APPEND_ARRAY(_blocks, sel.GetBlockArray());
    _blocks.Sort(wxblockint_sort_topleft_bottomright);
    Reset();
}

wxBlockIntSelectionIterator::wxBlockIntSelectionIterator(const wxArrayBlockInt& blocks, wxBLOCKINT_SELITER_Type type) {
    _type = type;
    WX_APPEND_ARRAY(_blocks, blocks);
    _blocks.Sort(wxblockint_sort_topleft_bottomright);
    Reset();
}

void wxBlockIntSelectionIterator::Reset() {
    _block_index = -1;
    _pt = wxPoint2DInt(0, 0);
}

bool wxBlockIntSelectionIterator::GetNext(wxBlockInt& block) {
    wxCHECK_MSG(_type == wxBLOCKINT_SELITER_BLOCK, false, wxT("wrong selection type"));
    if (_block_index + 1 < int(_blocks.GetCount())) {
        ++_block_index;
        block = _blocks[_block_index];
        return true;
    }

    return false;
}

bool wxBlockIntSelectionIterator::GetNext(wxPoint2DInt& pt) {
    wxCHECK_MSG(_type == wxBLOCKINT_SELITER_POINT, false, wxT("wrong selection type"));
    if ((_blocks.GetCount() < 1u) || (_block_index >= int(_blocks.GetCount()))) return false;

    // first time here
    if (_block_index < 0) {
        _block_index = 0;
        pt = _pt = _blocks[_block_index].GetLeftTop();
        return true;
    }

    // at end of block swap to new one
    if (_pt == _blocks[_block_index].GetRightBottom()) {
        ++_block_index;
        if (int(_blocks.GetCount()) > _block_index) {
            pt = _pt = _blocks[_block_index].GetLeftTop();
            return true;
        } else  // past end nothing more to check
            return false;
    }
    // at end of col, down to next row
    if (_pt._x == _blocks[_block_index].GetRight()) {
        _pt._x = _blocks[_block_index]._x1;
        _pt._y++;

        pt = _pt;
        return true;
    }

    // increment the col
    _pt._x++;
    pt = _pt;

    return true;
}

bool wxBlockIntSelectionIterator::IsInSelection(const wxPoint2DInt& pt) const {
    int n, count = _blocks.GetCount();
    for (n = 0; n < count; n++) {
        if (_blocks[n].Contains(pt)) return true;
    }
    return false;
}

//=============================================================================
// wxBlockDoubleSelectionIterator - iterates through a wxBlockDoubleSelection
//=============================================================================

wxBlockDoubleSelectionIterator::wxBlockDoubleSelectionIterator(const wxBlockDoubleSelection& sel) {
    WX_APPEND_ARRAY(_blocks, sel.GetBlockArray());
    _blocks.Sort(wxblockdouble_sort_topleft_bottomright);
    Reset();
}

wxBlockDoubleSelectionIterator::wxBlockDoubleSelectionIterator(const wxArrayBlockDouble& blocks) {
    WX_APPEND_ARRAY(_blocks, blocks);
    _blocks.Sort(wxblockdouble_sort_topleft_bottomright);
    Reset();
}

void wxBlockDoubleSelectionIterator::Reset() {
    _block_index = 0;
}

bool wxBlockDoubleSelectionIterator::GetNext(wxBlockDouble& block) {
    if (_block_index < _blocks.GetCount()) {
        block = _blocks[_block_index];
        _block_index++;
        return true;
    }

    return false;
}

bool wxBlockDoubleSelectionIterator::IsInSelection(const wxPoint2DDouble& pt) const {
    int n, count = _blocks.GetCount();
    for (n = 0; n < count; n++) {
        if (_blocks[n].Contains(pt)) return true;
    }
    return false;
}

// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// ============================================================================
// Unit testing, sortof

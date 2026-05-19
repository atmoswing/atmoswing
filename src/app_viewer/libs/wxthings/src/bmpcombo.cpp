/////////////////////////////////////////////////////////////////////////////
// Name:        wxBmpComboBox
// Purpose:     A wxComboBox type button for bitmaps and strings
// Author:      John Labenski
// Modified by:
// Created:     11/05/2002
// RCS-ID:
// Copyright:   (c) John Labenki
// Licence:     wxWidgets licence
/////////////////////////////////////////////////////////////////////////////

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP

#include "wx/bitmap.h"
#include "wx/control.h"
#include "wx/dc.h"
#include "wx/dcclient.h"
#include "wx/menu.h"
#include "wx/scrolwin.h"
#include "wx/settings.h"

#endif  // WX_PRECOMP

#include "wx/things/bmpcombo.h"

#if wxUSE_POPUPWIN

#define BORDER 4

// ============================================================================
// wxBmpComboPopupChild
// ============================================================================
IMPLEMENT_ABSTRACT_CLASS(wxBmpComboPopupChild, wxScrolledWindow)

BEGIN_EVENT_TABLE(wxBmpComboPopupChild, wxScrolledWindow)
EVT_PAINT(wxBmpComboPopupChild::OnPaint)
EVT_MOUSE_EVENTS(wxBmpComboPopupChild::OnMouse)
EVT_KEY_DOWN(wxBmpComboPopupChild::OnKeyDown)
END_EVENT_TABLE()

wxBmpComboPopupChild::wxBmpComboPopupChild(wxWindow* parent, wxBmpComboBox* owner)
    : wxScrolledWindow(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER | wxHSCROLL | wxVSCROLL) {
    _bmpCombo = owner;
    _last_selection = -1;
    SetBackgroundColour(_bmpCombo->GetBackgroundColour());
}

void wxBmpComboPopupChild::OnPaint(wxPaintEvent& WXUNUSED(event)) {
    wxPaintDC dc(this);
    PrepareDC(dc);
    // dc.SetBackground(*wxTheBrushList->FindOrCreateBrush(GetBackgroundColour(), wxBRUSHSTYLE_SOLID));
    // dc.Clear();

    dc.SetFont(_bmpCombo->GetFont());

    int y = 0, dy = _bmpCombo->GetItemSize().y;
    wxPoint origin = dc.GetDeviceOrigin();
    wxSize clientSize = GetClientSize();

    for (int n = 0; n < _bmpCombo->GetCount(); n++) {
        if (y + dy > -origin.y) {
            dc.SetDeviceOrigin(origin.x, origin.y + y + 1);
            _bmpCombo->DrawItem(dc, n);
        }

        y += dy;
        if (y > -origin.y + clientSize.y) break;
    }

    dc.SetDeviceOrigin(0, 0);
    PrepareDC(dc);  // reset back

    if (_bmpCombo->GetSelection() >= 0) {
        if (_last_selection < 0) _last_selection = _bmpCombo->GetSelection();

        DrawSelection(_last_selection, dc);
    }
}

void wxBmpComboPopupChild::OnMouse(wxMouseEvent& event) {
    wxPoint mouse = event.GetPosition();
    CalcUnscrolledPosition(mouse.x, mouse.y, &mouse.x, &mouse.y);

    // wxPrintf(wxT("bmpcombo mouse %d %d\n"), mouse.x, mouse.y); fflush(stdout);

    // Get selection from mouse pos, force valid
    int sel = _bmpCombo->GetItemSize().y != 0 ? mouse.y / _bmpCombo->GetItemSize().y : -1;
    if (sel < 0)
        sel = 0;
    else if (sel >= _bmpCombo->GetCount())
        sel = _bmpCombo->GetCount() - 1;

    if (event.LeftDown()) {
        // quickly show user what they selected before hiding it
        if (sel != _last_selection) {
            wxClientDC dc(this);
            PrepareDC(dc);
            if (_last_selection >= 0) DrawSelection(_last_selection, dc);
            if (sel >= 0) DrawSelection(sel, dc);

            _last_selection = sel;
        }

        _bmpCombo->SetSelection(sel, true);
        _bmpCombo->HidePopup();
        return;
    }
}

void wxBmpComboPopupChild::OnKeyDown(wxKeyEvent& event) {
    int sel = _last_selection;

    switch (event.GetKeyCode()) {
        case WXK_ESCAPE: {
            _bmpCombo->HidePopup();
            return;
        }
        case WXK_RETURN: {
            _bmpCombo->SetSelection(sel, true);
            _bmpCombo->HidePopup();
            return;
        }
        case WXK_UP:
            sel--;
            break;
        case WXK_DOWN:
            sel++;
            break;
        default:
            event.Skip(true);
            return;
    }

    if (sel < 0) sel = 0;
    if (sel >= _bmpCombo->GetCount()) sel = _bmpCombo->GetCount() - 1;

    if (sel != _last_selection) {
        wxClientDC dc(this);
        PrepareDC(dc);
        if (_last_selection >= 0) DrawSelection(_last_selection, dc);

        if (sel >= 0) DrawSelection(sel, dc);

        _last_selection = sel;
    }
}

void wxBmpComboPopupChild::DrawSelection(int n, wxDC& dc) {
    dc.SetBrush(*wxTRANSPARENT_BRUSH);
    dc.SetPen(*wxBLACK_PEN);
    dc.SetLogicalFunction(wxINVERT);
    int height = _bmpCombo->GetItemSize().y;
    dc.DrawRectangle(0, wxMax(0, height * n - 1), GetClientSize().x, height + 2);
    dc.SetLogicalFunction(wxCOPY);
}

// ==========================================================================
// wxBmpComboLabel - the main "window" to the left of the dropdown button
// ==========================================================================
IMPLEMENT_ABSTRACT_CLASS(wxBmpComboLabel, wxWindow)

BEGIN_EVENT_TABLE(wxBmpComboLabel, wxWindow)
EVT_PAINT(wxBmpComboLabel::OnPaint)
EVT_CHAR(wxBmpComboLabel::OnChar)
END_EVENT_TABLE()

void wxBmpComboLabel::OnChar(wxKeyEvent& event) {
    switch (event.GetKeyCode()) {
        case WXK_UP:
            _bmpCombo->SetNextSelection(false, true);
            break;
        case WXK_DOWN:
            _bmpCombo->SetNextSelection(true, true);
            break;
        default:
            break;
    }
}

void wxBmpComboLabel::OnPaint(wxPaintEvent& WXUNUSED(event)) {
    wxPaintDC dc(this);
    dc.SetFont(_bmpCombo->GetFont());
    // dc.SetBackground(*wxTheBrushList->FindOrCreateBrush(GetBackgroundColour(), wxBRUSHSTYLE_SOLID));
    // dc.Clear();
    dc.SetBrush(*wxTheBrushList->FindOrCreateBrush(GetBackgroundColour(), wxBRUSHSTYLE_SOLID));
    dc.SetPen(*wxTRANSPARENT_PEN);
    dc.DrawRectangle(wxRect(wxPoint(0, 0), GetClientSize()));

    const int sel = _bmpCombo->GetSelection();
    if ((sel >= 0) && (sel < _bmpCombo->GetCount())) _bmpCombo->DrawItem(dc, sel);
}

// ============================================================================
// wxBmpComboBox
// ============================================================================
IMPLEMENT_DYNAMIC_CLASS(wxBmpComboBox, DropDownBase)

BEGIN_EVENT_TABLE(wxBmpComboBox, DropDownBase)
EVT_SIZE(wxBmpComboBox::OnSize)
END_EVENT_TABLE()

wxBmpComboBox::~wxBmpComboBox() {
    while (_bitmaps.GetCount() > 0u) {
        wxBitmap* bmp = (wxBitmap*)_bitmaps.Item(0);
        _bitmaps.RemoveAt(0);
        delete bmp;
    }
}

void wxBmpComboBox::Init() {
    _labelWin = NULL;
    _frozen = true;
    _selection = 0;
    _win_border = 0;
    _label_style = wxBMPCOMBO_LEFT;
}

bool wxBmpComboBox::Create(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style,
                           const wxValidator& val, const wxString& name) {
    if (!DropDownBase::Create(parent, id, pos, size, wxNO_BORDER | wxCLIP_CHILDREN, val, name)) return false;

    _labelWin = new wxBmpComboLabel(this);
    _win_border = _labelWin->GetSize().x - _labelWin->GetClientSize().x;

    SetBackgroundColour(*wxWHITE);

    _frozen = false;
    CalcLayout();

    wxSize bestSize = DoGetBestSize();
    SetSize(wxSize(size.x < 0 ? bestSize.x : size.x, size.y < 0 ? bestSize.y : size.y));

    return SetButtonStyle(style);
}

#define BMPCOMBO_LABEL_MASK (wxBMPCOMBO_LEFT | wxBMPCOMBO_RIGHT)

bool wxBmpComboBox::SetButtonStyle(long style) {
    style &= BMPCOMBO_LABEL_MASK;  // strip off extras

    int n_styles = 0;
    if (style & wxBMPCOMBO_LEFT) n_styles++;
    if (style & wxBMPCOMBO_RIGHT) n_styles++;
    wxCHECK_MSG(n_styles < 2, false, wxT("Only one wxBmpComboBox label position allowed"));
    if (n_styles < 1) style |= (_label_style & BMPCOMBO_LABEL_MASK);

    _label_style = style;

    _labelWin->Refresh(true);

    return true;
}

void wxBmpComboBox::OnSize(wxSizeEvent& event) {
    event.Skip();

    if (!_labelWin || !_dropdownButton) return;

    wxSize size = GetClientSize();
    // wxPrintf(wxT("ComboOnSize %d %d\n"), size.x, size.y);
    int width = size.x - ((wxWindow*)_dropdownButton)->GetSize().x;
    _labelWin->SetSize(0, 0, width, size.y);
}

void wxBmpComboBox::DoSetSize(int x, int y, int width, int height, int sizeFlags) {
    /*
        wxSize curSize( GetSize() );

        if (width == -1)
            width = curSize.GetWidth();
        if (height == -1)
            height = curSize.GetHeight();
    */
    DropDownBase::DoSetSize(x, y, width, height, sizeFlags);
    /*
        width = width - ((wxWindow*)_dropdownButton)->GetSize().x;
        _labelWin->SetSize(0, 0, width, height);
    */
}

wxSize wxBmpComboBox::DoGetBestSize() const {
    if (GetCount() == 0) return DropDownBase::DoGetBestSize();

    wxSize size(0, 0);
    size.x = _labelSize.x + _bitmapSize.x + (_labelSize.x != 0 ? BORDER * 2 : 0);
    size.y = wxMax(_labelSize.y, _bitmapSize.y) + _win_border;

    size.x += _win_border + DROPDOWN_DROP_WIDTH;
    if (size.y < DROPDOWN_DROP_HEIGHT) size.y = DROPDOWN_DROP_HEIGHT;

    return size;
}

int wxBmpComboBox::DoGetBestDropHeight(int max_height) {
    int count = GetCount();
    if (count < 1) return -1;

    // add one for drawing selection rect
    return wxMin(_itemSize.y * count + _win_border + 1, max_height);
}

bool wxBmpComboBox::DoShowPopup() {
    if (_popupWin) {
        wxBmpComboPopupChild* popChild = new wxBmpComboPopupChild(_popupWin, this);
        _popupWin->SetChild(popChild);

        if (popChild) {
            popChild->_last_selection = GetSelection();
            int count = GetCount();
            int scr_pos = _selection > 0 ? _selection * _itemSize.y - 1 : 0;
            if (_popupWin->GetClientSize().GetHeight() >= _itemSize.y * count + 1) scr_pos = 0;
            popChild->SetScrollbars(1, 1, _itemSize.x, _itemSize.y * count + 1, 0, scr_pos);
        }
    }

    return DropDownBase::DoShowPopup();
}

void wxBmpComboBox::HidePopup() {
    DropDownBase::HidePopup();

    // FIXME - MSW destroys the sunken border of labelWin when in toolbar
    //         a refresh doesn't help
}

void wxBmpComboBox::Thaw() {
    _frozen = false;
    CalcLayout();
    if (_labelWin) _labelWin->Refresh();
}

void wxBmpComboBox::CalcLayout() {
    if (_frozen) return;

    int height = 0, width = 0;
    _itemSize = _labelSize = _bitmapSize = wxSize(0, 0);
    int count = GetCount();
    wxBitmap bmp;

    for (int n = 0; n < count; n++) {
        bmp = GetItemBitmap(n);
        if (bmp.Ok()) {
            width = bmp.GetWidth();
            height = bmp.GetHeight();

            if (width > _bitmapSize.x) _bitmapSize.x = width;
            if (height > _bitmapSize.y) _bitmapSize.y = height;
        }
        if (!_labels[n].IsEmpty()) {
            GetTextExtent(_labels[n], &width, &height);

            if (width > _labelSize.x) _labelSize.x = width;
            if (height > _labelSize.y) _labelSize.y = height;
        }
    }

    _itemSize.x = _labelSize.x + _bitmapSize.x + _win_border;
    _itemSize.y = wxMax(_labelSize.y, _bitmapSize.y) + _win_border;
}

void wxBmpComboBox::CalcLabelBitmapPos(int n, const wxSize& area, wxPoint& labelPos, wxPoint& bitmapPos) const {
    labelPos = bitmapPos = wxPoint(0, 0);

    int bw = 0, bh = 0;
    int lw = 0, lh = 0;

    if (GetItemBitmap(n).Ok()) {
        bw = GetItemBitmap(n).GetWidth();
        bh = GetItemBitmap(n).GetHeight();
    }
    if (!_labels[n].IsEmpty()) {
        GetTextExtent(_labels[n], &lw, &lh);
    }

    if (_bitmapSize.x == 0)  // There aren't any bitmaps, left align label
    {
        labelPos = wxPoint(BORDER, (area.y - lh) / 2);
    } else if (_labelSize.x == 0)  // There aren't any labels, center bitmap
    {
        bitmapPos = wxPoint((area.x - bw) / 2, (area.y - bh) / 2);
    } else if ((_label_style & wxBMPCOMBO_RIGHT) != 0) {
        labelPos = wxPoint(_bitmapSize.x + BORDER, (area.y - lh) / 2);
        bitmapPos = wxPoint((_bitmapSize.x - bw) / 2, (area.y - bh) / 2);
    } else  // if ((_label_style & wxBMPCOMBO_LEFT) != 0)
    {
        labelPos = wxPoint(BORDER, (area.y - lh) / 2);
        bitmapPos = wxPoint(BORDER * 2 + _labelSize.x + (area.x - BORDER * 2 - _labelSize.x - bw) / 2,
                            (area.y - bh) / 2);
    }
}

void wxBmpComboBox::DrawItem(wxDC& dc, int n) const {
    wxSize itemSize(GetItemSize());  //((wxWindow*)GetLabelWindow())->GetClientSize().x, dy);

    wxPoint labelPos, bitmapPos;
    CalcLabelBitmapPos(n, itemSize, labelPos, bitmapPos);

    if (GetItemBitmap(n).Ok()) dc.DrawBitmap(GetItemBitmap(n), bitmapPos.x, bitmapPos.y, true);
    if (!GetLabel(n).IsEmpty()) dc.DrawText(GetLabel(n), labelPos.x, labelPos.y);
}

int wxBmpComboBox::Append(const wxString& label, const wxBitmap& bitmap) {
    _labels.Add(label);
    _bitmaps.Add(new wxBitmap(bitmap));
    CalcLayout();
    return GetCount() - 1;
}

int wxBmpComboBox::Insert(const wxString& label, const wxBitmap& bitmap, unsigned int n) {
    wxCHECK_MSG(int(n) < GetCount(), wxNOT_FOUND, wxT("invalid index"));

    _labels.Insert(label, n);
    _bitmaps.Insert(new wxBitmap(bitmap), n);
    CalcLayout();
    return n;
}

void wxBmpComboBox::Clear() {
    _labels.Clear();
    while (_bitmaps.GetCount() > 0u) {
        wxBitmap* bmp = (wxBitmap*)_bitmaps.Item(0);
        _bitmaps.RemoveAt(0);
        delete bmp;
    }
    CalcLayout();
}

void wxBmpComboBox::Delete(unsigned int n, unsigned int count) {
    wxCHECK_RET(int(n + count) <= GetCount(), wxT("invalid index"));

    for (unsigned int i = 0; i < count; i++) {
        _labels.RemoveAt(n);
        wxBitmap* bmp = (wxBitmap*)_bitmaps.Item(n);
        _bitmaps.RemoveAt(n);
        delete bmp;
    }
    CalcLayout();
}

wxString wxBmpComboBox::GetLabel(int n) const {
    wxCHECK_MSG((n >= 0) && (n < GetCount()), wxEmptyString, wxT("invalid index"));
    return _labels[n];
}

wxBitmap wxBmpComboBox::GetItemBitmap(int n) const {
    wxCHECK_MSG((n >= 0) && (n < GetCount()), wxNullBitmap, wxT("invalid index"));
    return *(wxBitmap*)_bitmaps.Item(n);
}

void wxBmpComboBox::SetSelection(int n, bool send_event) {
    wxCHECK_RET((n >= 0) && (n < GetCount()), wxT("invalid index"));
    _selection = n;
    _labelWin->Refresh(true);

    if (send_event) {
        wxCommandEvent event(wxEVT_COMMAND_COMBOBOX_SELECTED, GetId());
        event.SetInt(_selection);
        event.SetEventObject(this);
        GetEventHandler()->ProcessEvent(event);
    }
}

void wxBmpComboBox::SetNextSelection(bool foward, bool send_event) {
    const int count = GetCount();
    if (count == 0) return;

    int sel = _selection;

    if (foward) {
        if ((sel < 0) || (sel == count - 1))
            sel = 0;
        else
            sel++;
    } else {
        if (sel <= 0)
            sel = count - 1;
        else
            sel--;
    }

    SetSelection(sel, send_event);
}

void wxBmpComboBox::SetLabel(int n, const wxString& label) {
    wxCHECK_RET((n >= 0) && (n < GetCount()), wxT("invalid index"));
    _labels[n] = label;
    CalcLayout();

    if (n == _selection) _labelWin->Refresh(false);
}

void wxBmpComboBox::SetItemBitmap(int n, const wxBitmap& bitmap) {
    wxCHECK_RET((n >= 0) && (n < GetCount()), wxT("invalid index"));
    *((wxBitmap*)_bitmaps.Item(n)) = bitmap;
    CalcLayout();

    if (n == _selection) _labelWin->Refresh(false);
}

bool wxBmpComboBox::SetBackgroundColour(const wxColour& colour) {
    // not a failure for wx 2.5.x since InheritAttributes calls this
    // from wxWindow::Create
    if (_labelWin) {
        _labelWin->SetBackgroundColour(colour);
        _labelWin->Refresh();
    }
    return DropDownBase::SetBackgroundColour(colour);
}

#endif  // wxUSE_POPUPWIN

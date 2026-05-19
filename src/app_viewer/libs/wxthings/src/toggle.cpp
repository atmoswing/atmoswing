/////////////////////////////////////////////////////////////////////////////
// Name:        wxCustomButton based on wxCustomToggleCtrl.cpp
// Purpose:     a toggle button
// Author:      Bruce Phillips
// Modified by: John Labenski
// Created:     11/05/2002
// RCS-ID:
// Copyright:   (c) Bruce Phillips, John Labenki
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
#include "wx/settings.h"
#include "wx/timer.h"

#endif  // WX_PRECOMP

#include "wx/image.h"
#include "wx/renderer.h"
#include "wx/tglbtn.h"
#include "wx/things/toggle.h"

// ==========================================================================
// wxCustomButton
// ==========================================================================
IMPLEMENT_DYNAMIC_CLASS(wxCustomButton, wxControl)

BEGIN_EVENT_TABLE(wxCustomButton, wxControl)
EVT_MOUSE_EVENTS(wxCustomButton::OnMouseEvents)
EVT_PAINT(wxCustomButton::OnPaint)
EVT_TIMER(wxID_ANY, wxCustomButton::OnTimer)
EVT_SIZE(wxCustomButton::OnSize)
END_EVENT_TABLE()

wxCustomButton::~wxCustomButton() {
    if (HasCapture()) ReleaseMouse();
    if (_timer) delete _timer;
}

void wxCustomButton::Init() {
    _focused = false;
    _labelMargin = wxSize(4, 4);
    _bitmapMargin = wxSize(2, 2);
    _down = 0;
    _timer = NULL;
    _eventType = 0;
    _button_style = wxCUSTBUT_TOGGLE | wxCUSTBUT_BOTTOM;
}

bool wxCustomButton::Create(wxWindow* parent, wxWindowID id, const wxString& label, const wxBitmap& bitmap,
                            const wxPoint& pos, const wxSize& size_, long style, const wxValidator& val,
                            const wxString& name) {
    _labelString = label;
    if (bitmap.Ok()) _bmpLabel = bitmap;
    wxSize bestSize = DoGetBestSize_(parent);
    wxSize size(size_.x < 0 ? bestSize.x : size_.x, size_.y < 0 ? bestSize.y : size_.y);

    // SetInitialSize(size);

    if (!wxControl::Create(parent, id, pos, size, wxNO_BORDER | wxCLIP_CHILDREN, val, name)) return false;

    wxControl::SetBackgroundColour(parent->GetBackgroundColour());
    wxControl::SetForegroundColour(parent->GetForegroundColour());
    wxControl::SetFont(parent->GetFont());

    if (!SetButtonStyle(style)) return false;

    // SetBestSize(size);

    CalcLayout(true);
    return true;
}

void wxCustomButton::SetValue(bool depressed) {
    wxCHECK_RET(!(_button_style & wxCUSTBUT_NOTOGGLE), wxT("can't set button state"));
    _down = depressed ? 1 : 0;
    Refresh(false);
}

bool wxCustomButton::SetButtonStyle(long style) {
    int n_styles = 0;
    if ((style & wxCUSTBUT_LEFT) != 0) n_styles++;
    if ((style & wxCUSTBUT_RIGHT) != 0) n_styles++;
    if ((style & wxCUSTBUT_TOP) != 0) n_styles++;
    if ((style & wxCUSTBUT_BOTTOM) != 0) n_styles++;
    wxCHECK_MSG(n_styles < 2, false, wxT("Only one wxCustomButton label position allowed"));

    n_styles = 0;
    if ((style & wxCUSTBUT_NOTOGGLE) != 0) n_styles++;
    if ((style & wxCUSTBUT_BUTTON) != 0) n_styles++;
    if ((style & wxCUSTBUT_TOGGLE) != 0) n_styles++;
    if ((style & wxCUSTBUT_BUT_DCLICK_TOG) != 0) n_styles++;
    if ((style & wxCUSTBUT_TOG_DCLICK_BUT) != 0) n_styles++;
    wxCHECK_MSG(n_styles < 2, false, wxT("Only one wxCustomButton style allowed"));

    _button_style = style;

    if ((_button_style & wxCUSTBUT_BUTTON) != 0) _down = 0;

    CalcLayout(true);
    return true;
}

void wxCustomButton::SetLabel(const wxString& label) {
    _labelString = label;
    InvalidateBestSize();
    CalcLayout(true);
}

// sequence of events in GTK is up, dclick, up.

void wxCustomButton::OnMouseEvents(wxMouseEvent& event) {
    if (_button_style & wxCUSTBUT_NOTOGGLE) return;

    if (event.LeftDown() || event.RightDown()) {
        if (!HasCapture()) CaptureMouse();  // keep depressed until up

        _down++;
        Redraw();
    } else if (event.LeftDClick() || event.RightDClick()) {
        _down++;  // GTK eats second down event
        Redraw();
    } else if (event.LeftUp()) {
        if (HasCapture()) ReleaseMouse();

        _eventType = wxEVT_LEFT_UP;

        if (wxRect(wxPoint(0, 0), GetSize()).Contains(event.GetPosition())) {
            if ((_button_style & wxCUSTBUT_BUTTON) && (_down > 0)) {
                _down = 0;
                Redraw();
                SendEvent();
                return;
            } else {
                if (!_timer) {
                    _timer = new wxTimer(this, _down + 1);
                    _timer->Start(200, true);
                } else {
                    _eventType = wxEVT_LEFT_DCLICK;
                }

                if ((_button_style & wxCUSTBUT_TOGGLE) && (_button_style & wxCUSTBUT_TOG_DCLICK_BUT)) _down++;
            }
        }

        Redraw();
    } else if (event.RightUp()) {
        if (HasCapture()) ReleaseMouse();

        _eventType = wxEVT_RIGHT_UP;

        if (wxRect(wxPoint(0, 0), GetSize()).Contains(event.GetPosition())) {
            if ((_button_style & wxCUSTBUT_BUTTON) && (_down > 0)) {
                _down = 0;
                Redraw();
                SendEvent();
                return;
            } else {
                _down++;

                if (!_timer) {
                    _timer = new wxTimer(this, _down);
                    _timer->Start(250, true);
                } else {
                    _eventType = wxEVT_RIGHT_DCLICK;
                }
            }
        }

        Redraw();
    } else if (event.Entering()) {
        _focused = true;
        if ((event.LeftIsDown() || event.RightIsDown()) && HasCapture()) _down++;

        Redraw();
    } else if (event.Leaving()) {
        _focused = false;
        if ((event.LeftIsDown() || event.RightIsDown()) && HasCapture()) _down--;

        Redraw();
    }
}

void wxCustomButton::OnTimer(wxTimerEvent& event) {
    _timer->Stop();
    delete _timer;
    _timer = NULL;

    // Clean up the button presses
    // FIXME - GTK eats second left down for a DClick, who know about the others?
    if (_button_style & wxCUSTBUT_BUTTON) {
        _down = 0;
    } else if (_button_style & wxCUSTBUT_TOGGLE) {
        if (_eventType == wxEVT_LEFT_UP)
            _down = event.GetId() % 2 ? 0 : 1;
        else
            _down = event.GetId() % 2 ? 1 : 0;
    } else if (_button_style & wxCUSTBUT_BUT_DCLICK_TOG) {
        if (_eventType == wxEVT_LEFT_DCLICK)
            _down = event.GetId() % 2 ? 0 : 1;
        else
            _down = event.GetId() % 2 ? 1 : 0;
    } else if (_button_style & wxCUSTBUT_TOG_DCLICK_BUT) {
        if (_eventType == wxEVT_LEFT_UP)
            _down = event.GetId() % 2 ? 0 : 1;
        else
            _down = event.GetId() % 2 ? 1 : 0;
    }

    Refresh(false);
    SendEvent();
}

void wxCustomButton::SendEvent() {
    if (((_button_style & wxCUSTBUT_TOGGLE) && (_eventType == wxEVT_LEFT_UP)) ||
        ((_button_style & wxCUSTBUT_BUT_DCLICK_TOG) && (_eventType == wxEVT_LEFT_DCLICK)) ||
        ((_button_style & wxCUSTBUT_TOG_DCLICK_BUT) && (_eventType == wxEVT_LEFT_UP))) {
        wxCommandEvent eventOut(wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, GetId());
        eventOut.SetInt(_down % 2 ? 1 : 0);
        eventOut.SetExtraLong(_eventType);
        eventOut.SetEventObject(this);
        GetEventHandler()->ProcessEvent(eventOut);
    } else {
        wxCommandEvent eventOut(wxEVT_COMMAND_BUTTON_CLICKED, GetId());
        eventOut.SetInt(0);
        eventOut.SetExtraLong(_eventType);
        eventOut.SetEventObject(this);
        GetEventHandler()->ProcessEvent(eventOut);
    }
}

wxBitmap wxCustomButton::CreateBitmapDisabled(const wxBitmap& bitmap) const {
    wxCHECK_MSG(bitmap.Ok(), wxNullBitmap, wxT("invalid bitmap"));

    unsigned char br = GetBackgroundColour().Red();
    unsigned char bg = GetBackgroundColour().Green();
    unsigned char bb = GetBackgroundColour().Blue();

    wxImage image = bitmap.ConvertToImage();
    int pos, width = image.GetWidth(), height = image.GetHeight();
    unsigned char* img_data = image.GetData();

    for (int j = 0; j < height; j++) {
        for (int i = j % 2; i < width; i += 2) {
            pos = (j * width + i) * 3;
            img_data[pos] = br;
            img_data[pos + 1] = bg;
            img_data[pos + 2] = bb;
        }
    }

    return wxBitmap(image);

    /*      // FIXME why bother creating focused wxCustomButton's bitmap
            wxImage imgFoc = bitmap.ConvertToImage();

            bool mask = false;
            unsigned char mr=0, mg=0, mb=0;
            if (img.HasMask())
            {
                mask = true;
                mr = imgDis.GetMaskRed();
                mg = imgDis.GetMaskGreen();
                mb = imgDis.GetMaskBlue();
            }
            unsigned char *r, *g, *b;
            unsigned char *focData = imgFoc.GetData();
            r = imgFoc.GetData();
            g = imgFoc.GetData() + 1;
            b = imgFoc.GetData() + 2;
            for (int j=0; j<h; j++)
            {
                for (int i=0; i<w; i++)
                {
                    if ((!mask || ((*r!=mr)&&(*b!=mb)&&(*g!=mg))) &&
                        ((*r<236)&&(*b<236)&&(*g<236)))
                    {
                        *r += 20; *g += 20; *b += 20;
                    }
                    r += 3; g += 3; b += 3;
                }
            }
            _bmpFocus = wxBitmap(imgFoc);
    */
}

void wxCustomButton::SetBitmapLabel(const wxBitmap& bitmap) {
    _bmpLabel = bitmap;
    InvalidateBestSize();
    CalcLayout(true);
}

void wxCustomButton::OnPaint(wxPaintEvent& WXUNUSED(event)) {
    wxPaintDC dc(this);
    Paint(dc);
}

void wxCustomButton::Redraw() {
    wxClientDC dc(this);
    Paint(dc);
}

void wxCustomButton::Paint(wxDC& dc) {
    int w, h;
    GetSize(&w, &h);

    wxColour foreColour = GetForegroundColour();
    wxColour backColour = GetBackgroundColour();

    if (_focused) {
        backColour.Set(wxMin(backColour.Red() + 20, 255), wxMin(backColour.Green() + 20, 255),
                       wxMin(backColour.Blue() + 20, 255));
    }

    wxBitmap bitmap;

    if (IsEnabled()) {
        if (GetValue() && _bmpSelected.Ok())
            bitmap = _bmpSelected;
        else if (_focused && _bmpFocus.Ok())
            bitmap = _bmpFocus;
        else if (_bmpLabel.Ok())
            bitmap = _bmpLabel;
    } else {
        // try to create disabled if it doesn't exist
        if (!_bmpDisabled.Ok() && _bmpLabel.Ok()) _bmpDisabled = CreateBitmapDisabled(_bmpLabel);

        if (_bmpDisabled.Ok())
            bitmap = _bmpDisabled;
        else if (_bmpLabel.Ok())
            bitmap = _bmpLabel;

        foreColour = wxSystemSettings::GetColour(wxSYS_COLOUR_GRAYTEXT);
    }

#if wxCHECK_VERSION(2, 8, 0)

    // wxCONTROL_DISABLED
    // flags may have the wxCONTROL_PRESSED, wxCONTROL_CURRENT or wxCONTROL_ISDEFAULT

    int ren_flags = 0;
    if (GetValue()) ren_flags |= wxCONTROL_PRESSED;
    if (_focused) ren_flags |= wxCONTROL_CURRENT;
    if (!IsEnabled()) ren_flags |= wxCONTROL_DISABLED;

    wxRendererNative::Get().DrawPushButton(this, dc, wxRect(0, 0, w, h), ren_flags);

#else

    wxBrush brush(backColour, wxBRUSHSTYLE_SOLID);
    dc.SetBackground(brush);
    dc.SetBrush(brush);
    dc.SetPen(*wxTRANSPARENT_PEN);

    dc.DrawRectangle(0, 0, w, h);

#endif  // !wxCHECK_VERSION(2, 8, 0)

    if (bitmap.Ok()) dc.DrawBitmap(bitmap, _bitmapPos.x, _bitmapPos.y, true);

    if (!GetLabel().IsEmpty()) {
        dc.SetFont(GetFont());
        dc.SetTextBackground(backColour);
        dc.SetTextForeground(foreColour);
        dc.DrawText(GetLabel(), _labelPos.x, _labelPos.y);
    }

#if !wxCHECK_VERSION(2, 8, 0)
    if (GetValue())  // draw sunken border
    {
        dc.SetPen(*wxGREY_PEN);
        dc.DrawLine(0, h - 1, 0, 0);
        dc.DrawLine(0, 0, w, 0);
        dc.SetPen(*wxWHITE_PEN);
        dc.DrawLine(w - 1, 1, w - 1, h - 1);
        dc.DrawLine(w - 1, h - 1, 0, h - 1);
        dc.SetPen(*wxBLACK_PEN);
        dc.DrawLine(1, h - 2, 1, 1);
        dc.DrawLine(1, 1, w - 1, 1);
    } else if (((_button_style & wxCUSTBUT_FLAT) == 0) || _focused)  // draw raised border
    {
        dc.SetPen(*wxWHITE_PEN);
        dc.DrawLine(0, h - 2, 0, 0);
        dc.DrawLine(0, 0, w - 1, 0);
        dc.SetPen(*wxBLACK_PEN);
        dc.DrawLine(w - 1, 0, w - 1, h - 1);
        dc.DrawLine(w - 1, h - 1, -1, h - 1);
        dc.SetPen(*wxGREY_PEN);
        dc.DrawLine(2, h - 2, w - 2, h - 2);
        dc.DrawLine(w - 2, h - 2, w - 2, 1);
    }
#endif  // !wxCHECK_VERSION(2, 8, 0)

    dc.SetBackground(wxNullBrush);
    dc.SetBrush(wxNullBrush);
    dc.SetPen(wxNullPen);
}

void wxCustomButton::OnSize(wxSizeEvent& event) {
    CalcLayout(true);
    event.Skip();
}

void wxCustomButton::SetMargins(const wxSize& margin, bool fit) {
    _labelMargin = margin;
    _bitmapMargin = margin;
    if (fit) SetSize(DoGetBestSize());
    CalcLayout(true);
}

void wxCustomButton::SetLabelMargin(const wxSize& margin, bool fit) {
    _labelMargin = margin;
    CalcLayout(true);
    if (fit) SetSize(DoGetBestSize());
}

void wxCustomButton::SetBitmapMargin(const wxSize& margin, bool fit) {
    _bitmapMargin = margin;
    CalcLayout(true);
    if (fit) SetSize(DoGetBestSize());
}

wxSize wxCustomButton::DoGetBestSize() const {
    return DoGetBestSize_((wxWindow*)this);
}

wxSize wxCustomButton::DoGetBestSize_(wxWindow* win) const {
    //((wxWindow*)this)->InvalidateBestSize();

    int lw = 0, lh = 0;
    int bw = 0, bh = 0;
    bool has_bitmap = _bmpLabel.Ok();
    bool has_label = !_labelString.IsEmpty();

    if (has_label) {
        win->GetTextExtent(_labelString, &lw, &lh);
        lw += 2 * _labelMargin.x;
        lh += 2 * _labelMargin.y;
    }
    if (has_bitmap) {
        bw = _bmpLabel.GetWidth() + 2 * _bitmapMargin.x;
        bh = _bmpLabel.GetHeight() + 2 * _bitmapMargin.y;
    }

    if (((_button_style & wxCUSTBUT_LEFT) != 0) || ((_button_style & wxCUSTBUT_RIGHT) != 0)) {
        int h = (bh > lh) ? bh : lh;
        if (has_bitmap && has_label) lw -= wxMin(_labelMargin.x, _bitmapMargin.x);

        return wxSize(lw + bw, h);
    }

    int w = (bw > lw) ? bw : lw;
    if (has_bitmap && has_label) lh -= wxMin(_labelMargin.y, _bitmapMargin.y);

    return wxSize(w, lh + bh);
}

void wxCustomButton::CalcLayout(bool refresh) {
    int w, h;
    GetSize(&w, &h);

    int bw = 0, bh = 0;
    int lw = 0, lh = 0;
    bool has_bitmap = _bmpLabel.Ok();
    bool has_label = !GetLabel().IsEmpty();

    if (has_bitmap)  // assume they're all the same size
    {
        bw = _bmpLabel.GetWidth();
        bh = _bmpLabel.GetHeight();
    }

    if (has_label) {
        GetTextExtent(GetLabel(), &lw, &lh);
    }

    // Center the label or bitmap if only one or the other
    if (!has_bitmap) {
        _bitmapPos = wxPoint(0, 0);
        _labelPos = wxPoint((w - lw) / 2, (h - lh) / 2);
    } else if (!has_label) {
        _bitmapPos = wxPoint((w - bw) / 2, (h - bh) / 2);
        _labelPos = wxPoint(0, 0);
    } else if ((_button_style & wxCUSTBUT_LEFT) != 0) {
        int mid_margin = wxMax(_labelMargin.x, _bitmapMargin.x);
        _labelPos = wxPoint((w - (bw + lw + _labelMargin.x + _bitmapMargin.x + mid_margin)) / 2 + _labelMargin.x,
                            (h - lh) / 2);
        _bitmapPos = wxPoint(_labelPos.x + lw + mid_margin, (h - bh) / 2);
    } else if ((_button_style & wxCUSTBUT_RIGHT) != 0) {
        int mid_margin = wxMax(_labelMargin.x, _bitmapMargin.x);
        _bitmapPos = wxPoint((w - (bw + lw + _labelMargin.x + _bitmapMargin.x + mid_margin)) / 2 + _bitmapMargin.x,
                             (h - bh) / 2);
        _labelPos = wxPoint(_bitmapPos.x + bw + mid_margin, (h - lh) / 2);
    } else if ((_button_style & wxCUSTBUT_TOP) != 0) {
        int mid_margin = wxMax(_labelMargin.y, _bitmapMargin.y);
        _labelPos = wxPoint((w - lw) / 2,
                            (h - (bh + lh + _labelMargin.y + _bitmapMargin.y + mid_margin)) / 2 + _labelMargin.y);
        _bitmapPos = wxPoint((w - bw) / 2, _labelPos.y + lh + mid_margin);
    } else  // if ((_button_style & wxCUSTBUT_BOTTOM) != 0)  DEFAULT
    {
        int mid_margin = wxMax(_labelMargin.y, _bitmapMargin.y);
        _bitmapPos = wxPoint((w - bw) / 2,
                             (h - (bh + lh + _labelMargin.y + _bitmapMargin.y + mid_margin)) / 2 + _bitmapMargin.y);
        _labelPos = wxPoint((w - lw) / 2, _bitmapPos.y + bh + mid_margin);
    }

    if (refresh) Refresh(false);
}

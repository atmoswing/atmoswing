/////////////////////////////////////////////////////////////////////////////
// Name:        led.cpp
// Purpose:
// Author:      Joachim Buermann
// Id:          $Id$
// Copyright:   (c) 2001 Joachim Buermann
/////////////////////////////////////////////////////////////////////////////

#include "led.h"

#include <wx/wxprec.h>

#include "asBitmaps.h"
#include "asIncludes.h"

BEGIN_EVENT_TABLE(awxLed, wxWindow)
EVT_ERASE_BACKGROUND(awxLed::OnErase)
EVT_PAINT(awxLed::OnPaint)
EVT_SIZE(awxLed::OnSizeEvent)
END_EVENT_TABLE()

awxLed::awxLed(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, awxLedColour color, long style,
               int timerInterval)
    : wxWindow(parent, id, pos, size, wxNO_FULL_REPAINT_ON_RESIZE | style),
      _bitmap(new wxBitmap(16 * g_ppiScaleDc, 16 * g_ppiScaleDc)),
      _state(awxLED_OFF),
      _blink(0),
      _x(0),
      _y(0),
      _timerInterval(timerInterval),
      _on(false) {
    int imgSize = 16 * g_ppiScaleDc;
    _timer = new BlinkTimer(this);
    _icons[awxLED_OFF] = asBitmaps::Get(asBitmaps::ID_BULLETS::BULLET_WHITE);
    _icons[awxLED_ON] = asBitmaps::Get(asBitmaps::ID_BULLETS::BULLET_WHITE);
    SetInitialSize(wxSize(imgSize, imgSize));
    SetMinSize(wxSize(imgSize, imgSize));
    SetColour(color);
}

awxLed::~awxLed() {
    if (_timer) {
        _timer->Stop();
        delete _timer;
    }
    delete _bitmap;
}

void awxLed::Blink() {
    _blink ^= 1;
    Redraw();
}

void awxLed::DrawOnBitmap() {
    /*
    wxSize s = GetClientSize();
    if ((_bitmap->GetWidth() != s.GetWidth()) || (_bitmap->GetHeight() != s.GetHeight())) {
        _bitmap->Create(s.x, s.y);
    }*/
    wxMemoryDC dc;
    dc.SelectObject(*_bitmap);

    wxBrush brush(_parent->GetBackgroundColour(), wxBRUSHSTYLE_SOLID);
    dc.SetBackground(brush);
    dc.Clear();

    if (_state == awxLED_BLINK)
        dc.DrawBitmap(_icons[_blink], _x, _y, true);
    else
        dc.DrawBitmap(_icons[_state & 1], _x, _y, true);

    dc.SelectObject(wxNullBitmap);
}

void awxLed::SetColour(awxLedColour colour) {
    // if(_icons[awxLED_ON]) delete _icons[awxLED_ON];
    switch (colour) {
        case awxLED_LUCID:
            _icons[awxLED_ON] = asBitmaps::Get(asBitmaps::ID_BULLETS::BULLET_WHITE);
            break;
        case awxLED_GREEN:
            _icons[awxLED_ON] = asBitmaps::Get(asBitmaps::ID_BULLETS::BULLET_GREEN);
            break;
        case awxLED_YELLOW:
            _icons[awxLED_ON] = asBitmaps::Get(asBitmaps::ID_BULLETS::BULLET_YELLOW);
            break;
        default:
            _icons[awxLED_ON] = asBitmaps::Get(asBitmaps::ID_BULLETS::BULLET_RED);
    }
}

void awxLed::SetState(awxLedState state) {
    _state = state;
    if (_timer->IsRunning()) {
        _timer->Stop();
    }
    if (_state == awxLED_BLINK) {
        _timer->Start(_timerInterval);
    }
    Redraw();
}

void awxLed::SetOn(awxLedColour colour, awxLedState state) {
    _onColour = colour;
    _onState = state;
}

void awxLed::SetOff(awxLedColour colour, awxLedState state) {
    _offColour = colour;
    _offState = state;
}

void awxLed::TurnOn(bool on) {
    _on = on;
    if (on) {
        SetColour(_onColour);
        SetState(_onState);
    } else {
        SetColour(_offColour);
        SetState(_offState);
    }
}

void awxLed::TurnOff() {
    _on = false;
    SetColour(_offColour);
    SetState(_offState);
}

void awxLed::Toggle() {
    TurnOn(!_on);
}

bool awxLed::IsOn() {
    return _on;
}

void awxLed::SetTimerInterval(unsigned int timerInterval) {
    _timerInterval = timerInterval;
    SetState(_state);
}

unsigned int awxLed::GetTimerInterval() {
    return _timerInterval;
}

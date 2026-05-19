/////////////////////////////////////////////////////////////////////////////
// Name:        led.h
// Purpose:
// Author:      Joachim Buermann
// Id:          $Id$
// Copyright:   (c) 2001 Joachim Buermann
/////////////////////////////////////////////////////////////////////////////

#ifndef __LED_H
#define __LED_H

#ifndef WX_PRECOMP

#include "wx/wx.h"

#endif

enum awxLedState {
    awxLED_OFF = 0,
    awxLED_ON,
    awxLED_BLINK
};

enum awxLedColour {
    awxLED_LUCID = 0,
    awxLED_RED,
    awxLED_GREEN,
    awxLED_YELLOW
};

class BlinkTimer;

class awxLed : public wxWindow {
  protected:
    // bitmap for double buffering
    wxBitmap* _bitmap;
    wxBitmap _icons[2];
    awxLedState _state;
    BlinkTimer* _timer;
    int _blink;
    int _x;
    int _y;
    unsigned int _timerInterval;
    bool _on;
    awxLedState _onState;
    awxLedState _offState;
    awxLedColour _onColour;
    awxLedColour _offColour;

  protected:
    // protected member functions
    void DrawOnBitmap();

  public:
    awxLed(wxWindow* parent, wxWindowID id, const wxPoint& pos = wxPoint(0, 0), const wxSize& size = wxSize(16, 16),
           // red LED is default
           awxLedColour color = awxLED_RED, long style = 0, int timerInterval = 500);

    ~awxLed() override;

    void Blink();

    void OnErase(wxEraseEvent&) {
        Redraw();
    };

    void OnPaint(wxPaintEvent&) {
        wxPaintDC dc(this);
        dc.DrawBitmap(*_bitmap, 0, 0, false);
    };

    void OnSizeEvent(wxSizeEvent& event) {
        wxSize size = event.GetSize();
        _x = (size.GetX() - _icons[0].GetWidth()) >> 1;
        _y = (size.GetY() - _icons[0].GetHeight()) >> 1;
        if (_x < 0) _x = 0;
        if (_y < 0) _y = 0;
        DrawOnBitmap();
    };

    void Redraw() {
        wxClientDC dc(this);
        DrawOnBitmap();
        dc.DrawBitmap(*_bitmap, 0, 0, false);
    };

    void SetTimerInterval(unsigned int timerInterval);

    unsigned int GetTimerInterval();

    void SetColour(awxLedColour colour);

    void SetState(awxLedState state);

    void SetOn(awxLedColour colour, awxLedState state = awxLED_ON);

    void SetOff(awxLedColour colour, awxLedState state = awxLED_ON);

    void TurnOn(bool on = true);

    void TurnOff();

    void Toggle();

    bool IsOn();

    DECLARE_EVENT_TABLE()
};

class BlinkTimer : public wxTimer {
  protected:
    awxLed* _led;

  public:
    BlinkTimer(awxLed* led)
        : wxTimer() {
        _led = led;
    };

    void Notify() {
        _led->Blink();
    };
};

#endif

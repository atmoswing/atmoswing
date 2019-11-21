/////////////////////////////////////////////////////////////////////////////
// Name:        led.cpp
// Purpose:
// Author:      Joachim Buermann
// Id:          $Id$
// Copyright:   (c) 2001 Joachim Buermann
/////////////////////////////////////////////////////////////////////////////

#include "led.h"

#include <wx/wxprec.h>

#include "asIncludes.h"
#include "images.h"

BEGIN_EVENT_TABLE(awxLed, wxWindow)
EVT_ERASE_BACKGROUND(awxLed::OnErase) EVT_PAINT(awxLed::OnPaint) EVT_SIZE(awxLed::OnSizeEvent) END_EVENT_TABLE()

    awxLed::awxLed(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, awxLedColour color,
                   long style, int timerInterval)
    : wxWindow(parent, id, pos, size, wxNO_FULL_REPAINT_ON_RESIZE | style),
      m_bitmap(new wxBitmap(16 * g_ppiScaleDc, 16 * g_ppiScaleDc)),
      m_state(awxLED_OFF),
      m_blink(0),
      m_x(0),
      m_y(0),
      m_timerInterval(timerInterval),
      m_on(false) {
    int imgSize = 16 * g_ppiScaleDc;
    m_timer = new BlinkTimer(this);
    m_icons[awxLED_OFF] = *_img_bullet_white;
    m_icons[awxLED_ON] = *_img_bullet_white;
    SetInitialSize(wxSize(imgSize, imgSize));
    SetMinSize(wxSize(imgSize, imgSize));
    SetColour(color);
}

awxLed::~awxLed() {
    if (m_timer) {
        m_timer->Stop();
        delete m_timer;
    }
    delete m_bitmap;
}

void awxLed::Blink() {
    m_blink ^= 1;
    Redraw();
}

void awxLed::DrawOnBitmap() {
    /*
    wxSize s = GetClientSize();
    if ((m_bitmap->GetWidth() != s.GetWidth()) || (m_bitmap->GetHeight() != s.GetHeight())) {
        m_bitmap->Create(s.x, s.y);
    }*/
    wxMemoryDC dc;
    dc.SelectObject(*m_bitmap);

    wxBrush brush(m_parent->GetBackgroundColour(), wxBRUSHSTYLE_SOLID);
    dc.SetBackground(brush);
    dc.Clear();

    if (m_state == awxLED_BLINK)
        dc.DrawBitmap(m_icons[m_blink], m_x, m_y, true);
    else
        dc.DrawBitmap(m_icons[m_state & 1], m_x, m_y, true);

    dc.SelectObject(wxNullBitmap);
}

void awxLed::SetColour(awxLedColour colour) {
    // if(m_icons[awxLED_ON]) delete m_icons[awxLED_ON];
    switch (colour) {
        case awxLED_LUCID:
            m_icons[awxLED_ON] = *_img_bullet_white;
            break;
        case awxLED_GREEN:
            m_icons[awxLED_ON] = *_img_bullet_green;
            break;
        case awxLED_YELLOW:
            m_icons[awxLED_ON] = *_img_bullet_yellow;
            break;
        default:
            m_icons[awxLED_ON] = *_img_bullet_red;
    }
}

void awxLed::SetState(awxLedState state) {
    m_state = state;
    if (m_timer->IsRunning()) {
        m_timer->Stop();
    }
    if (m_state == awxLED_BLINK) {
        m_timer->Start(m_timerInterval);
    }
    Redraw();
}

void awxLed::SetOn(awxLedColour colour, awxLedState state) {
    m_onColour = colour;
    m_onState = state;
}

void awxLed::SetOff(awxLedColour colour, awxLedState state) {
    m_offColour = colour;
    m_offState = state;
}

void awxLed::TurnOn(bool on) {
    m_on = on;
    if (on) {
        SetColour(m_onColour);
        SetState(m_onState);
    } else {
        SetColour(m_offColour);
        SetState(m_offState);
    }
}

void awxLed::TurnOff() {
    m_on = false;
    SetColour(m_offColour);
    SetState(m_offState);
}

void awxLed::Toggle() {
    TurnOn(!m_on);
}

bool awxLed::IsOn() {
    return m_on;
}

void awxLed::SetTimerInterval(unsigned int timerInterval) {
    m_timerInterval = timerInterval;
    SetState(m_state);
}

unsigned int awxLed::GetTimerInterval() {
    return m_timerInterval;
}

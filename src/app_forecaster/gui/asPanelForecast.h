/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef AS_PANEL_FORECAST
#define AS_PANEL_FORECAST

#include <wx/awx/led.h>

#include "AtmoSwingForecasterGui.h"
#include "asBitmaps.h"

class asPanelsManagerForecasts;
class asBatchForecasts;

class asPanelForecast : public asPanelForecastVirtual {
  public:
    /**
     * Constructor.
     *
     * @param parent The parent window.
     * @param batch The batch of forecasts.
     */
    explicit asPanelForecast(wxWindow* parent, asBatchForecasts* batch);

    /**
     * Layout the panel.
     *
     * @return True if done.
     */
    bool Layout() override;

    /**
     * Check if the forecast file exists.
     */
    void CheckFileExists();

    /**
     * Set the content of the tooltip. It provides the description of the forecast.
     *
     * @param filePath The path to the forecast file.
     */
    void SetTooTipContent(const wxString& filePath);

    /**
     * Access the LED.
     * 
     * @return The LED pointer.
     */
    awxLed* GetLed() const {
        return m_led;
    }

    /**
     * Set the panel manager.
     * 
     * @param panelManager The panel manager.
     */
    void SetPanelsManager(asPanelsManagerForecasts* panelManager) {
        m_panelsManager = panelManager;
    }

    /**
     * Access the forecast parameters file name.
     * 
     * @return The file name.
     */
    wxString GetParametersFileName() const {
        return m_textParametersFileName->GetLabel();
    }

    /**
     * Set the forecast parameters file name.
     * 
     * @param val The file name.
     */
    void SetParametersFileName(const wxString& val) {
        m_textParametersFileName->SetLabel(val);
        CheckFileExists();
    }

  protected:
    wxWindow* m_parentFrame; /**< The parent frame. */
    awxLed* m_led; /**< The LED. */
    asBatchForecasts* m_batchForecasts; /**< The batch of forecasts. */

    /**
     * Close the panel.
     *
     * @param event The command event.
     */
    void ClosePanel(wxCommandEvent& event) override;

    /**
     * Edit the forecast file path.
     *
     * @param event The command event.
     */
    void OnEditForecastFile(wxCommandEvent& event) override;

    /**
     * Show the details of the forecast on a styled text control.
     *
     * @param event The command event.
     */
    void OnDetailsForecastFile(wxCommandEvent& event) override;

  private:
    asPanelsManagerForecasts* m_panelsManager; /**< The panels manager. */
};

#endif

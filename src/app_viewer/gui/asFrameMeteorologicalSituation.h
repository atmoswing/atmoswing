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
 * The Original Software is AtmoSwing. The Initial Developer of the
 * Original Software is Pascal Horton of the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2014 Pascal Horton, Terr@num.
 */

#ifndef ASFRAMEMETEOSITUATION_H
#define ASFRAMEMETEOSITUATION_H

#include "AtmoswingViewerGui.h"
#include "asIncludes.h"
#include "asLogWindow.h"
#include "asPanelSidebarGisLayers.h"
#include "asPanelSidebarMeteoSituation.h"
#include "wx/dnd.h"
#include <wx/process.h>
#include "vroomgis.h"

const int asID_MENU_POPUP_LAYER_METEO = wxID_HIGHEST + 3 + 50;


/** Implementing vroomDropFiles */
class asFrameMeteorologicalSituation;
class vroomDropFilesMeteoSituation : public wxFileDropTarget
{
private:
    asFrameMeteorologicalSituation * m_LoaderFrame;

public:
    vroomDropFilesMeteoSituation(asFrameMeteorologicalSituation * parent);
    virtual bool OnDropFiles(wxCoord x, wxCoord y,
                             const wxArrayString & filenames);
};


class asFrameMeteorologicalSituation : public asFrameMeteorologicalSituationVirtual
{
public:
    asFrameMeteorologicalSituation(wxWindow* parent, wxWindowID id=asWINDOW_MAIN);
    virtual ~asFrameMeteorologicalSituation();

    void OnInit();
    bool OpenLayers (const wxArrayString & names);

    void OnToolZoomIn (wxCommandEvent & event);
    void OnToolZoomOut (wxCommandEvent & event);
    void OnToolPan (wxCommandEvent & event);
    void OnKeyDown(wxKeyEvent & event);
    void OnKeyUp(wxKeyEvent & event);
    void OnToolAction (wxCommandEvent & event);
    void OnToolZoomToFit (wxCommandEvent & event);
    void OnQuit( wxCommandEvent& event );


    vrLayerManager *GetLayerManager()
    {
        return m_LayerManager;
    }

    void SetLayerManager(vrLayerManager* layerManager)
    {
        m_LayerManager = layerManager;
    }

    vrViewerLayerManager *GetViewerLayerManager()
    {
        return m_ViewerLayerManager;
    }

    void SetViewerLayerManager(vrViewerLayerManager* viewerLayerManager)
    {
        m_ViewerLayerManager = viewerLayerManager;
    }

    vrViewerDisplay *GetViewerDisplay()
    {
        return m_DisplayCtrl;
    }

    void SetViewerDisplay(vrViewerDisplay* viewerDisplay)
    {
        m_DisplayCtrl = viewerDisplay;
    }


protected:
    // vroomgis
    vrLayerManager *m_LayerManager;
    vrViewerLayerManager *m_ViewerLayerManager;
    vrViewerDisplay *m_DisplayCtrl;
    wxKeyboardState m_KeyBoardState;
    asPanelSidebarGisLayers *m_PanelSidebarGisLayers;
    asPanelSidebarMeteoSituation *m_PanelSidebarMeteoSituation;

    void OnOpenLayer( wxCommandEvent & event );
    void OnCloseLayer( wxCommandEvent & event );
    void OnMoveLayer( wxCommandEvent & event );
    void OnToolDisplayValue( wxCommandEvent & event );
    void OnPresetSelection (wxCommandEvent & event);
    void ReloadViewerLayerManager( );
    #if defined (__WIN32__)
        wxCriticalSection m_CritSectionViewerLayerManager;
    #endif


    virtual void OnRightClick( wxMouseEvent& event )
    {
        event.Skip();
    }

private:
    
    DECLARE_EVENT_TABLE()
};

#endif // ASFRAMEMETEOSITUATION_H

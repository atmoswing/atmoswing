#ifndef __asFramePredictors__
#define __asFramePredictors__

#include "AtmoswingViewerGui.h"
#include "asIncludes.h"
#include "asForecastManager.h"
#include "asPredictorsViewer.h"
#include "vroomgis.h"
#include "wx/dnd.h"

const int asID_SET_SYNCRO_MODE	= wxID_HIGHEST + 1;
const int asID_SIGHT_TOOL		= wxID_HIGHEST + 2;


/** Implementing vroomDropFiles */
class asFramePredictors;
class vroomDropFilesPredictors : public wxFileDropTarget
{
private:
    asFramePredictors * m_LoaderFrame;

public:
    vroomDropFilesPredictors(asFramePredictors * parent);
    virtual bool OnDropFiles(wxCoord x, wxCoord y,
                             const wxArrayString & filenames);
};


/** Implementing asFramePredictors */
class asFramePredictors : public asFramePredictorsVirtual
{
public:
    /** Constructor */
    asFramePredictors( wxWindow* parent, int selectedForecast, asForecastManager *forecastManager, wxWindowID id=asWINDOW_PREDICTORS );
    ~asFramePredictors();

    void Init();
    void InitExtent();
    bool OpenLayers (const wxArrayString & names);
    void OpenDefaultLayers ();

protected:
    wxKeyboardState m_KeyBoardState;

    void OnSwitchRight( wxCommandEvent& event );
    void OnSwitchLeft( wxCommandEvent& event );
    void OpenFramePreferences( wxCommandEvent& event );
    void OnPredictorSelectionChange( wxCommandEvent& event );
    void OnForecastChange( wxCommandEvent& event );
    void OnTargetDateChange( wxCommandEvent& event );
    void OnAnalogDateChange( wxCommandEvent& event );
    void OnOpenLayer(wxCommandEvent & event);
    void OnCloseLayer(wxCommandEvent & event);
    void OnSyncroToolSwitch(wxCommandEvent & event);
    void OnToolZoomIn (wxCommandEvent & event);
    void OnToolZoomOut (wxCommandEvent & event);
    void OnToolPan (wxCommandEvent & event);
    void OnToolSight (wxCommandEvent & event);
    void OnToolZoomToFit (wxCommandEvent & event);
    void OnToolAction (wxCommandEvent & event);
    void OnKeyDown (wxKeyEvent & event);
    void OnKeyUp (wxKeyEvent & event);
    void UpdateLayers();
    void ReloadViewerLayerManagerLeft();
    void ReloadViewerLayerManagerRight();

    virtual void OnRightClick( wxMouseEvent& event )
	{
	    event.Skip();
    }

private:
    asForecastManager* m_ForecastManager;
    asPredictorsViewer* m_PredictorsViewer;
    asPredictorsManager* m_PredictorsManager;
    int m_SelectedForecast;
    int m_SelectedTargetDate;
    int m_SelectedAnalogDate;
    bool m_SyncroTool;
    bool m_DisplayPanelLeft;
    bool m_DisplayPanelRight;
	wxOverlay m_Overlay;
    #if defined (__WIN32__)
        wxCriticalSection m_CritSectionViewerLayerManager;
    #endif

	// vroomgis
	vrLayerManager *m_LayerManager;
	vrViewerTOCTree * m_TocCtrlLeft;
	vrViewerTOCTree * m_TocCtrlRight;
	vrViewerLayerManager *m_ViewerLayerManagerLeft;
	vrViewerLayerManager *m_ViewerLayerManagerRight;
	vrViewerDisplay *m_DisplayCtrlLeft;
	vrViewerDisplay *m_DisplayCtrlRight;

    DECLARE_EVENT_TABLE()
};

#endif // __asFramePredictors__

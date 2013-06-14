#include "asFramePredictors.h"

#include "asFramePreferences.h"
#if defined (__WIN32__)
    #include "asThreadsManager.h"
    #include "asThreadViewerLayerManagerReload.h"
    #include "asThreadViewerLayerManagerZoomIn.h"
    #include "asThreadViewerLayerManagerZoomOut.h"
#endif
#include "asResultsAnalogsForecast.h"
#include "asCatalogPredictorsArchive.h"
#include "asDataPredictorArchive.h"
#include "asPredictorsManager.h"
#include "asGeoAreaCompositeGrid.h"
#include "img_bullets.h"
#include "img_toolbar.h"
#include <wx/colour.h>
#include "vrrender.h"
#include "vrlayervector.h"
#include <wx/statline.h>
#include <wx/app.h>
#include <wx/event.h>

BEGIN_EVENT_TABLE(asFramePredictors, wxFrame)
/*    EVT_MENU(wxID_EXIT,  asFramePredictors::OnQuit)
    EVT_MENU(wxID_ABOUT, asFramePredictors::OnAbout)*/
	EVT_MENU(wxID_OPEN, asFramePredictors::OnOpenLayer)
	EVT_MENU(wxID_REMOVE, asFramePredictors::OnCloseLayer)
/*	EVT_MENU (wxID_INFO, asFramePredictors::OnShowLog)*/
	//EVT_MENU (wxID_DEFAULT, asFramePredictors::OnToolSelect)
	EVT_MENU (asID_ZOOM_IN, asFramePredictors::OnToolZoomIn)
	EVT_MENU (asID_ZOOM_OUT, asFramePredictors::OnToolZoomOut)
	EVT_MENU (asID_ZOOM_FIT, asFramePredictors::OnToolZoomToFit)
	EVT_MENU (asID_PAN, asFramePredictors::OnToolPan)
	EVT_MENU (asID_CROSS_MARKER, asFramePredictors::OnToolSight)
	EVT_MENU (asID_SET_SYNCRO_MODE, asFramePredictors::OnSyncroToolSwitch)
	//EVT_MENU (wxID_ZOOM_100, asFramePredictors::OnToolZoomToFit)

	EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOM, asFramePredictors::OnToolAction)
	EVT_COMMAND(wxID_ANY, vrEVT_TOOL_ZOOMOUT, asFramePredictors::OnToolAction)
	EVT_COMMAND(wxID_ANY, vrEVT_TOOL_PAN, asFramePredictors::OnToolAction)
	EVT_COMMAND(wxID_ANY, vrEVT_TOOL_SIGHT, asFramePredictors::OnToolAction)

END_EVENT_TABLE()

/* vroomDropFilesPredictors */

vroomDropFilesPredictors::vroomDropFilesPredictors(asFramePredictors * parent)
{
	wxASSERT(parent);
	m_LoaderFrame = parent;
}


bool vroomDropFilesPredictors::OnDropFiles(wxCoord x, wxCoord y,
								 const wxArrayString & filenames)
{
	if (filenames.GetCount() == 0) return false;

	m_LoaderFrame->OpenLayers(filenames);
	return true;
}


asFramePredictors::asFramePredictors( wxWindow* parent, int selectedForecast, asForecastManager *forecastManager, wxWindowID id )
:
asFramePredictorsVirtual( parent, id )
{
    m_ForecastManager = forecastManager;
    m_SelectedForecast = selectedForecast;
    m_SelectedTargetDate = -1;
    m_SelectedAnalogDate = -1;
    m_SyncroTool = true;
    m_DisplayPanelLeft = true;
    m_DisplayPanelRight = true;

    // Toolbar
	m_ToolBar->AddTool( asID_ZOOM_IN, wxT("Zoom in"), img_map_zoom_in, img_map_zoom_in, wxITEM_NORMAL, _("Zoom in"), _("Zoom in"), NULL );
	m_ToolBar->AddTool( asID_ZOOM_OUT, wxT("Zoom out"), img_map_zoom_out, img_map_zoom_out, wxITEM_NORMAL, _("Zoom out"), _("Zoom out"), NULL );
	m_ToolBar->AddTool( asID_PAN, wxT("Pan"), img_map_move, img_map_move, wxITEM_NORMAL, _("Pan the map"), _("Move the map by panning"), NULL );
	m_ToolBar->AddTool( asID_ZOOM_FIT, wxT("Fit"), img_map_fit, img_map_fit, wxITEM_NORMAL, _("Zoom to visible layers"), _("Zoom view to the full extent of all visible layers"), NULL );
	m_ToolBar->AddTool( asID_CROSS_MARKER, wxT("Marker overlay"), img_map_cross, img_map_cross, wxITEM_NORMAL, _("Display a cross marker overlay"), _("Display a cross marker overlay on both frames"), NULL );
	m_ToolBar->AddSeparator();
	m_ToolBar->AddTool( asID_EXPORT_PDF, wxT("Export to pdf"), img_generate_pdf, img_generate_pdf, wxITEM_NORMAL, _("Export to pdf"), _("Export to pdf"), NULL );
	m_ToolBar->AddTool( asID_PRINT, wxT("Print"), img_print, img_print, wxITEM_NORMAL, _("Print"), _("Print"), NULL );
	m_ToolBar->AddSeparator();
	m_ToolBar->AddTool( asID_PREFERENCES, wxT("Preferences"), img_preferences, img_preferences, wxITEM_NORMAL, _("Preferences"), _("Preferences"), NULL );
	m_ToolBar->Realize();

    // VroomGIS controls
	m_DisplayCtrlLeft = new vrViewerDisplay( m_PanelGISLeft, wxID_ANY, wxColour(255,255,255));
	m_DisplayCtrlRight = new vrViewerDisplay( m_PanelGISRight, wxID_ANY, wxColour(255,255,255));
	m_SizerGISLeft->Add( m_DisplayCtrlLeft, 1, wxEXPAND|wxALL, 0 );
	m_SizerGISRight->Add( m_DisplayCtrlRight, 1, wxEXPAND|wxALL, 0 );
	m_PanelGIS->Layout();
	m_TocCtrlLeft = new vrViewerTOCTree( m_ScrolledWindowOptions, wxID_ANY);
	m_TocCtrlRight = new vrViewerTOCTree( m_ScrolledWindowOptions, wxID_ANY);
	m_SizerScrolledWindow->Insert( 5, m_TocCtrlLeft->GetControl(), 1, wxEXPAND, 5 );
	m_SizerScrolledWindow->Add( m_TocCtrlRight->GetControl(), 1, wxEXPAND, 5 );
	m_SizerScrolledWindow->Fit(m_ScrolledWindowOptions);

	m_LayerManager = new vrLayerManager();
	m_ViewerLayerManagerLeft = new vrViewerLayerManager(m_LayerManager, this, m_DisplayCtrlLeft , m_TocCtrlLeft);
	m_ViewerLayerManagerRight = new vrViewerLayerManager(m_LayerManager, this, m_DisplayCtrlRight , m_TocCtrlRight);

    // Viewer
    m_PredictorsManager = new asPredictorsManager();
    m_PredictorsViewer = new asPredictorsViewer(this, m_LayerManager, m_PredictorsManager, m_ViewerLayerManagerLeft, m_ViewerLayerManagerRight, m_CheckListPredictors);

	// Menus
	m_MenuTools->AppendCheckItem(asID_SET_SYNCRO_MODE, "Syncronize tools",
							  "When set to true, browsing is syncronized on all display");
	m_MenuTools->Check(asID_SET_SYNCRO_MODE, m_SyncroTool);

    // Connect Events
	m_DisplayCtrlLeft->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( asFramePredictors::OnRightClick ), NULL, this );
	m_DisplayCtrlLeft->Connect( wxEVT_KEY_DOWN, wxKeyEventHandler( asFramePredictors::OnKeyDown ), NULL, this);
	m_DisplayCtrlLeft->Connect( wxEVT_KEY_UP, wxKeyEventHandler( asFramePredictors::OnKeyUp ), NULL, this);
    this->Connect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFramePredictors::OpenFramePreferences ) );

    // DND
	m_ScrolledWindowOptions->SetDropTarget(new vroomDropFilesPredictors(this));

	// Bitmap
	m_BpButtonSwitchRight->SetBitmapLabel(img_arrows_right);
	m_BpButtonSwitchLeft->SetBitmapLabel(img_arrows_left);

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFramePredictors::~asFramePredictors()
{
    // Disconnect Events
	m_DisplayCtrlLeft->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( asFramePredictors::OnRightClick ), NULL, this );
	m_DisplayCtrlLeft->Disconnect( wxEVT_KEY_DOWN, wxKeyEventHandler( asFramePredictors::OnKeyDown ), NULL, this);
	m_DisplayCtrlLeft->Disconnect( wxEVT_KEY_UP, wxKeyEventHandler( asFramePredictors::OnKeyUp ), NULL, this);
    this->Disconnect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFramePredictors::OpenFramePreferences ) );

    wxDELETE(m_LayerManager);
}

void asFramePredictors::Init()
{
    if (m_ForecastManager->GetModelsNb()>0)
    {
        // Forecast list
        wxArrayString arrayForecasts = m_ForecastManager->GetModelsNamesWxArray();
        m_ChoiceForecast->Set(arrayForecasts);
        m_ChoiceForecast->Select(m_SelectedForecast);

        m_SelectedTargetDate = 0;
        m_SelectedAnalogDate = 0;

        // Target dates list
        wxArrayString arrayDates = m_ForecastManager->GetLeadTimes(m_SelectedForecast);
        m_ChoiceTargetDates->Set(arrayDates);
        m_ChoiceTargetDates->Select(m_SelectedTargetDate);

        // Analog dates list
        asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(m_SelectedForecast);
        Array1DFloat analogDates = forecast->GetAnalogsDates(m_SelectedTargetDate);
        wxArrayString arrayAnalogDates;
        for (int i=0; i<analogDates.size(); i++)
        {
            arrayAnalogDates.Add(asTime::GetStringTime(analogDates[i], "DD.MM.YYYY"));
        }
        m_ChoiceAnalogDates->Set(arrayAnalogDates);
        m_ChoiceAnalogDates->Select(m_SelectedAnalogDate);
    }

    // Predictors checklist
//    m_CheckListPredictors

    // GIS
    InitExtent();
    OpenDefaultLayers();
}

void asFramePredictors::InitExtent()
{
    // Desired extent
	vrRealRect desiredExtent;
	desiredExtent.m_x = -20;
	desiredExtent.m_width = 40;
	desiredExtent.m_y = 60;
	desiredExtent.m_height = -30;

	m_ViewerLayerManagerLeft->InitializeExtent(desiredExtent);
	m_ViewerLayerManagerRight->InitializeExtent(desiredExtent);

}

void asFramePredictors::OnSwitchRight( wxCommandEvent& event )
{
    if (!m_DisplayPanelRight) return;

    Freeze();

    if (m_DisplayPanelLeft)
    {
        m_SizerGIS->Hide(m_PanelRight, true);
        m_DisplayPanelRight = false;
    }
    else
    {
        m_SizerGIS->Show(m_PanelLeft, true);
        m_SizerGIS->Show(m_PanelRight, true);
        m_DisplayPanelLeft = true;
        m_DisplayPanelRight = true;
    }

    m_SizerGIS->Fit(m_PanelGIS);
    Layout();
    Refresh();
    Thaw();
}

void asFramePredictors::OnSwitchLeft( wxCommandEvent& event )
{
    if (!m_DisplayPanelLeft) return;

    Freeze();

    if (m_DisplayPanelRight)
    {
        m_SizerGIS->Hide(m_PanelLeft, true);
        m_DisplayPanelLeft = false;
    }
    else
    {
        m_SizerGIS->Show(m_PanelLeft, true);
        m_SizerGIS->Show(m_PanelRight, true);
        m_DisplayPanelLeft = true;
        m_DisplayPanelRight = true;
    }

    m_SizerGIS->Fit(m_PanelGIS);
    Layout();
    Refresh();
    Thaw();
}

void asFramePredictors::OnPredictorSelectionChange( wxCommandEvent& event )
{
    UpdateLayers();
}

void asFramePredictors::OnForecastChange( wxCommandEvent& event )
{
    m_SelectedForecast = event.GetInt();

    // Target dates list
    wxArrayString arrayTargetDates = m_ForecastManager->GetLeadTimes(m_SelectedForecast);
    m_ChoiceTargetDates->Set(arrayTargetDates);
    if (arrayTargetDates.size()<=(unsigned)m_SelectedTargetDate)
    {
        m_SelectedTargetDate = 0;
    }
    m_ChoiceTargetDates->Set(arrayTargetDates);
    m_ChoiceTargetDates->Select(m_SelectedTargetDate);

    // Analog dates list
    asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(m_SelectedForecast);
    Array1DFloat analogDates = forecast->GetAnalogsDates(m_SelectedTargetDate);
    wxArrayString arrayAnalogDates;
    for (int i=0; i<analogDates.size(); i++)
    {
        arrayAnalogDates.Add(asTime::GetStringTime(analogDates[i], "DD.MM.YYYY"));
    }
    if (arrayAnalogDates.size()<=(unsigned)m_SelectedAnalogDate)
    {
        m_SelectedAnalogDate = 0;
    }
    m_ChoiceAnalogDates->Set(arrayAnalogDates);
    m_ChoiceAnalogDates->Select(m_SelectedAnalogDate);

    UpdateLayers();
}

void asFramePredictors::OnTargetDateChange( wxCommandEvent& event )
{
    m_SelectedTargetDate = event.GetInt();

    // Analog dates list
    asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(m_SelectedForecast);
    Array1DFloat analogDates = forecast->GetAnalogsDates(m_SelectedTargetDate);
    wxArrayString arrayAnalogDates;
    for (int i=0; i<analogDates.size(); i++)
    {
        arrayAnalogDates.Add(asTime::GetStringTime(analogDates[i], "DD.MM.YYYY"));
    }
    if (arrayAnalogDates.size()<=(unsigned)m_SelectedAnalogDate)
    {
        m_SelectedAnalogDate = 0;
    }
    m_ChoiceAnalogDates->Set(arrayAnalogDates);
    m_ChoiceAnalogDates->Select(m_SelectedAnalogDate);

    UpdateLayers();
}

void asFramePredictors::OnAnalogDateChange( wxCommandEvent& event )
{
    UpdateLayers();
}

void asFramePredictors::OpenFramePreferences( wxCommandEvent& event )
{
    asFramePreferences* frame = new asFramePreferences(this, asWINDOW_PREFERENCES);
    frame->Fit();
    frame->Show();
}

void asFramePredictors::OpenDefaultLayers()
{
    // Default paths
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString dirData = asConfig::GetDataDir()+"data"+DS;
    wxString layerContinentsFilePath = pConfig->Read("/GIS/LayerContinentsFilePath", dirData+"gis"+DS+"World"+DS+"continents.shp");
    wxString layerCountriesFilePath = pConfig->Read("/GIS/LayerCountriesFilePath", dirData+"gis"+DS+"World"+DS+"countries.shp");
    wxString layerLatLongFilePath = pConfig->Read("/GIS/LayerLatLongFilePath", dirData+"gis"+DS+"World"+DS+"latlong.shp");
    wxString layerGeogridFilePath = pConfig->Read("/GIS/LayerGeogridFilePath", dirData+"gis"+DS+"World"+DS+"geogrid.shp");

    // Try to open layers
    m_ViewerLayerManagerLeft->FreezeBegin();
    m_ViewerLayerManagerRight->FreezeBegin();
    vrLayer * layer;

    // Continents
    if (wxFileName::FileExists(layerContinentsFilePath))
    {
        if (m_LayerManager->Open(wxFileName(layerContinentsFilePath)))
        {
            long layerContinentsTransp = 50;
            layerContinentsTransp = pConfig->Read("/GIS/LayerContinentsTransp", layerContinentsTransp);
            long layerContinentsColor = (long)0x99999999;
            layerContinentsColor = pConfig->Read("/GIS/LayerContinentsColor", layerContinentsColor);
            wxColour colorContinents;
            colorContinents.SetRGB((wxUint32)layerContinentsColor);
            long layerContinentsSize = 1;
            layerContinentsSize = pConfig->Read("/GIS/LayerContinentsSize", layerContinentsSize);
            bool layerContinentsVisibility;
            pConfig->Read("/GIS/LayerContinentsVisibility", &layerContinentsVisibility, true);

            vrRenderVector* renderContinents1 = new vrRenderVector();
            renderContinents1->SetTransparency(layerContinentsTransp);
            renderContinents1->SetColorPen(colorContinents);
            renderContinents1->SetColorBrush(colorContinents);
            renderContinents1->SetBrushStyle(wxBRUSHSTYLE_SOLID);
            renderContinents1->SetSize(layerContinentsSize);
            vrRenderVector* renderContinents2 = new vrRenderVector();
            renderContinents2->SetTransparency(layerContinentsTransp);
            renderContinents2->SetColorPen(colorContinents);
            renderContinents2->SetColorBrush(colorContinents);
            renderContinents2->SetBrushStyle(wxBRUSHSTYLE_SOLID);
            renderContinents2->SetSize(layerContinentsSize);

            layer = m_LayerManager->GetLayer( wxFileName(layerContinentsFilePath));
            wxASSERT(layer);
            m_ViewerLayerManagerLeft->Add(-1, layer, renderContinents1, NULL, layerContinentsVisibility);
            m_ViewerLayerManagerRight->Add(-1, layer, renderContinents2, NULL, layerContinentsVisibility);
        }
        else
        {
            asLogWarning(wxString::Format(_("The Continents layer file %s cound not be opened."), layerContinentsFilePath.c_str()));
        }
    }
    else
    {
        asLogWarning(wxString::Format(_("The Continents layer file %s cound not be found."), layerContinentsFilePath.c_str()));
    }

    // Countries
    if (wxFileName::FileExists(layerCountriesFilePath))
    {
        if (m_LayerManager->Open(wxFileName(layerCountriesFilePath)))
        {
            long layerCountriesTransp = 0;
            layerCountriesTransp = pConfig->Read("/GIS/LayerCountriesTransp", layerCountriesTransp);
            long layerCountriesColor = (long)0x77999999;
            layerCountriesColor = pConfig->Read("/GIS/LayerCountriesColor", layerCountriesColor);
            wxColour colorCountries;
            colorCountries.SetRGB((wxUint32)layerCountriesColor);
            long layerCountriesSize = 1;
            layerCountriesSize = pConfig->Read("/GIS/LayerCountriesSize", layerCountriesSize);
            bool layerCountriesVisibility;
            pConfig->Read("/GIS/LayerCountriesVisibility", &layerCountriesVisibility, true);

            vrRenderVector* renderCountries1 = new vrRenderVector();
            renderCountries1->SetTransparency(layerCountriesTransp);
            renderCountries1->SetColorPen(colorCountries);
            renderCountries1->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderCountries1->SetSize(layerCountriesSize);
            vrRenderVector* renderCountries2 = new vrRenderVector();
            renderCountries2->SetTransparency(layerCountriesTransp);
            renderCountries2->SetColorPen(colorCountries);
            renderCountries2->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderCountries2->SetSize(layerCountriesSize);

            layer = m_LayerManager->GetLayer( wxFileName(layerCountriesFilePath));
            wxASSERT(layer);
            m_ViewerLayerManagerLeft->Add(-1, layer, renderCountries1, NULL, layerCountriesVisibility);
            m_ViewerLayerManagerRight->Add(-1, layer, renderCountries2, NULL, layerCountriesVisibility);
        }
        else
        {
            asLogWarning(wxString::Format(_("The Countries layer file %s cound not be opened."), layerCountriesFilePath.c_str()));
        }
    }
    else
    {
        asLogWarning(wxString::Format(_("The Countries layer file %s cound not be found."), layerCountriesFilePath.c_str()));
    }

    // LatLong
    if (wxFileName::FileExists(layerLatLongFilePath))
    {
        if (m_LayerManager->Open(wxFileName(layerLatLongFilePath)))
        {
            long layerLatLongTransp = 80;
            layerLatLongTransp = pConfig->Read("/GIS/LayerLatLongTransp", layerLatLongTransp);
            long layerLatLongColor = (long)0xff999999;
            layerLatLongColor = pConfig->Read("/GIS/LayerLatLongColor", layerLatLongColor);
            wxColour colorLatLong;
            colorLatLong.SetRGB((wxUint32)layerLatLongColor);
            long layerLatLongSize = 1;
            layerLatLongSize = pConfig->Read("/GIS/LayerLatLongSize", layerLatLongSize);
            bool layerLatLongVisibility;
            pConfig->Read("/GIS/LayerLatLongVisibility", &layerLatLongVisibility, true);

            vrRenderVector* renderLatLong1 = new vrRenderVector();
            renderLatLong1->SetTransparency(layerLatLongTransp);
            renderLatLong1->SetColorPen(colorLatLong);
            renderLatLong1->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderLatLong1->SetSize(layerLatLongSize);
            vrRenderVector* renderLatLong2 = new vrRenderVector();
            renderLatLong2->SetTransparency(layerLatLongTransp);
            renderLatLong2->SetColorPen(colorLatLong);
            renderLatLong2->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderLatLong2->SetSize(layerLatLongSize);

            layer = m_LayerManager->GetLayer( wxFileName(layerLatLongFilePath));
            wxASSERT(layer);
            m_ViewerLayerManagerLeft->Add(-1, layer, renderLatLong1, NULL, layerLatLongVisibility);
            m_ViewerLayerManagerRight->Add(-1, layer, renderLatLong2, NULL, layerLatLongVisibility);
        }
        else
        {
            asLogWarning(wxString::Format(_("The LatLong layer file %s cound not be opened."), layerLatLongFilePath.c_str()));
        }
    }
    else
    {
        asLogWarning(wxString::Format(_("The LatLong layer file %s cound not be found."), layerLatLongFilePath.c_str()));
    }

    // Geogrid
    if (wxFileName::FileExists(layerGeogridFilePath))
    {
        if (m_LayerManager->Open(wxFileName(layerGeogridFilePath)))
        {
            long layerGeogridTransp = 80;
            layerGeogridTransp = pConfig->Read("/GIS/LayerGeogridTransp", layerGeogridTransp);
            long layerGeogridColor = (long)0xff999999;
            layerGeogridColor = pConfig->Read("/GIS/LayerGeogridColor", layerGeogridColor);
            wxColour colorGeogrid;
            colorGeogrid.SetRGB((wxUint32)layerGeogridColor);
            long layerGeogridSize = 2;
            layerGeogridSize = pConfig->Read("/GIS/LayerGeogridSize", layerGeogridSize);
            bool layerGeogridVisibility;
            pConfig->Read("/GIS/LayerGeogridVisibility", &layerGeogridVisibility, false);

            vrRenderVector* renderGeogrid1 = new vrRenderVector();
            renderGeogrid1->SetTransparency(layerGeogridTransp);
            renderGeogrid1->SetColorPen(colorGeogrid);
            renderGeogrid1->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderGeogrid1->SetSize(layerGeogridSize);
            vrRenderVector* renderGeogrid2 = new vrRenderVector();
            renderGeogrid2->SetTransparency(layerGeogridTransp);
            renderGeogrid2->SetColorPen(colorGeogrid);
            renderGeogrid2->SetBrushStyle(wxBRUSHSTYLE_TRANSPARENT);
            renderGeogrid2->SetSize(layerGeogridSize);

            layer = m_LayerManager->GetLayer( wxFileName(layerGeogridFilePath));
            wxASSERT(layer);
            m_ViewerLayerManagerLeft->Add(-1, layer, renderGeogrid1, NULL, layerGeogridVisibility);
            m_ViewerLayerManagerRight->Add(-1, layer, renderGeogrid2, NULL, layerGeogridVisibility);
        }
        else
        {
            asLogWarning(wxString::Format(_("The Geogrid layer file %s cound not be opened."), layerGeogridFilePath.c_str()));
        }
    }
    else
    {
        asLogWarning(wxString::Format(_("The Geogrid layer file %s cound not be found."), layerGeogridFilePath.c_str()));
    }

	m_ViewerLayerManagerLeft->FreezeEnd();
	m_ViewerLayerManagerRight->FreezeEnd();

}

bool asFramePredictors::OpenLayers (const wxArrayString & names)
{
    // Open files
	for (unsigned int i = 0; i< names.GetCount(); i++)
	{
		if(!m_LayerManager->Open(wxFileName(names.Item(i))))
		{
		    asLogError(_("The layer could not be opened."));
		    return false;
		}
	}

    // Get files
    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Enter();
    #endif
	m_ViewerLayerManagerLeft->FreezeBegin();
	m_ViewerLayerManagerRight->FreezeBegin();
	for (unsigned int i = 0; i< names.GetCount(); i++)
	{
		vrLayer * layer = m_LayerManager->GetLayer( wxFileName(names.Item(i)));
		wxASSERT(layer);

		// Add files to the viewer
		m_ViewerLayerManagerLeft->Add(1, layer, NULL);
		m_ViewerLayerManagerRight->Add(1, layer, NULL);
	}
	m_ViewerLayerManagerLeft->FreezeEnd();
	m_ViewerLayerManagerRight->FreezeEnd();
	#if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Leave();
    #endif
	return true;

}

void asFramePredictors::OnOpenLayer(wxCommandEvent & event)
{
	vrDrivers drivers;
	wxFileDialog myFileDlg (this, _("Select GIS layers"),
							wxEmptyString,
							wxEmptyString,
							drivers.GetWildcards(),
                            wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_MULTIPLE | wxFD_CHANGE_DIR);

	wxArrayString pathsFileName;

    // Try to open files
	if(myFileDlg.ShowModal()==wxID_OK)
	{
		myFileDlg.GetPaths(pathsFileName);
		wxASSERT(pathsFileName.GetCount() > 0);

		OpenLayers(pathsFileName);
	}
}

void asFramePredictors::OnCloseLayer(wxCommandEvent & event)
{
    #if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Enter();
    #endif

	wxArrayString layersName;
	for (int i=0; i<m_ViewerLayerManagerLeft->GetCount(); i++)
    {
		vrRenderer * renderer = m_ViewerLayerManagerLeft->GetRenderer(i);
		wxASSERT(renderer);
		layersName.Add(renderer->GetLayer()->GetDisplayName().GetFullName());
	}

	if (layersName.IsEmpty())
    {
		asLogError("No layer opened, nothing to close");
		#if defined (__WIN32__)
            m_CritSectionViewerLayerManager.Leave();
        #endif
		return;
	}

	wxMultiChoiceDialog choiceDlg (this, "Select Layer(s) to close",
									 "Close layer(s)",
									 layersName);
	if (choiceDlg.ShowModal() != wxID_OK)
    {
        #if defined (__WIN32__)
            m_CritSectionViewerLayerManager.Leave();
        #endif
        return;
    }

	wxArrayInt layerToRemoveIndex = choiceDlg.GetSelections();
	if (layerToRemoveIndex.IsEmpty())
    {
		asLogWarning("Nothing selected, no layer will be closed");
		#if defined (__WIN32__)
            m_CritSectionViewerLayerManager.Leave();
        #endif
		return;
	}

	// Removing layer(s)
	m_ViewerLayerManagerLeft->FreezeBegin();
	m_ViewerLayerManagerRight->FreezeBegin();

	for (int j = (signed) layerToRemoveIndex.GetCount() -1; j >= 0 ; j--)
    {
		// Remove from viewer manager (TOC and Display)
		vrRenderer * rendererLeft = m_ViewerLayerManagerLeft->GetRenderer(layerToRemoveIndex.Item(j));
		vrLayer * layer = rendererLeft->GetLayer();
		wxASSERT(rendererLeft);
		m_ViewerLayerManagerLeft->Remove(rendererLeft);

		vrRenderer * rendererRight = m_ViewerLayerManagerRight->GetRenderer(layerToRemoveIndex.Item(j));
		wxASSERT(rendererRight);
		m_ViewerLayerManagerRight->Remove(rendererRight);

		// Close layer (not used anymore);
		m_LayerManager->Close(layer);
	}
	m_ViewerLayerManagerLeft->FreezeEnd();
	m_ViewerLayerManagerRight->FreezeEnd();
	#if defined (__WIN32__)
        m_CritSectionViewerLayerManager.Leave();
    #endif
}

void asFramePredictors::OnKeyDown(wxKeyEvent & event)
{
	m_KeyBoardState = wxKeyboardState(event.ControlDown(),
									  event.ShiftDown(),
									  event.AltDown(),
									  event.MetaDown());
	if (m_KeyBoardState.GetModifiers() != wxMOD_CMD)
    {
		event.Skip();
		return;
	}

	const vrDisplayTool * tool = m_DisplayCtrlLeft->GetTool();
	if (tool == NULL)
    {
		event.Skip();
		return;
	}

	if (tool->GetID() == wxID_ZOOM_IN)
    {
		m_DisplayCtrlLeft->SetToolZoomOut();
		m_DisplayCtrlRight->SetToolZoomOut();
	}
	event.Skip();
}

void asFramePredictors::OnKeyUp(wxKeyEvent & event)
{
	if (m_KeyBoardState.GetModifiers() != wxMOD_CMD) {
		event.Skip();
		return;
	}

	const vrDisplayTool * tool = m_DisplayCtrlLeft->GetTool();
	if (tool == NULL) {
		event.Skip();
		return;
	}

	if (tool->GetID() == wxID_ZOOM_OUT || tool->GetID() == wxID_ZOOM_IN) {
		m_DisplayCtrlLeft->SetToolZoom();
		m_DisplayCtrlRight->SetToolZoom();
	}
	event.Skip();
}

void asFramePredictors::OnSyncroToolSwitch(wxCommandEvent & event)
{
	m_SyncroTool = GetMenuBar()->IsChecked(asID_SET_SYNCRO_MODE);
}

void asFramePredictors::OnToolZoomIn (wxCommandEvent & event)
{
	m_DisplayCtrlLeft->SetToolZoom();
	m_DisplayCtrlRight->SetToolZoom();
}

void asFramePredictors::OnToolZoomOut (wxCommandEvent & event)
{
	m_DisplayCtrlLeft->SetToolZoomOut();
	m_DisplayCtrlRight->SetToolZoomOut();
}

void asFramePredictors::OnToolPan (wxCommandEvent & event)
{
	m_DisplayCtrlLeft->SetToolPan();
	m_DisplayCtrlRight->SetToolPan();
}

void asFramePredictors::OnToolSight (wxCommandEvent & event)
{
	m_DisplayCtrlLeft->SetToolSight();
	m_DisplayCtrlRight->SetToolSight();
}

void asFramePredictors::OnToolZoomToFit (wxCommandEvent & event)
{
    if (m_DisplayPanelLeft)
    {
        m_ViewerLayerManagerLeft->ZoomToFit(false);
        ReloadViewerLayerManagerLeft();
    }
    if (m_DisplayPanelRight)
    {
        m_ViewerLayerManagerRight->ZoomToFit(false);
        ReloadViewerLayerManagerRight();
    }
}

void asFramePredictors::OnToolAction (wxCommandEvent & event)
{
	vrDisplayToolMessage * msg = (vrDisplayToolMessage*)event.GetClientData();
	wxASSERT(msg);

	if(msg->m_EvtType == vrEVT_TOOL_ZOOM)
    {
		// getting rectangle
		vrCoordinate * coord = msg->m_ParentManager->GetDisplay()->GetCoordinate();
		wxASSERT(coord);

		// get real rectangle
		vrRealRect realRect;
		bool success = coord->ConvertFromPixels(msg->m_Rect, realRect);
		wxASSERT(success == true);

		// get fitted rectangle
		vrRealRect fittedRect = coord->GetRectFitted(realRect);
		wxASSERT(fittedRect.IsOk());

		if (m_SyncroTool == false)
		{
		    #if defined (__WIN32__)
                asThreadViewerLayerManagerZoomIn *thread = new asThreadViewerLayerManagerZoomIn(msg->m_ParentManager, &m_CritSectionViewerLayerManager, fittedRect);
                ThreadsManager().AddThread(thread);
            #else
                msg->m_ParentManager->Zoom(fittedRect);
            #endif
		}
		else
		{
            if (m_DisplayPanelLeft)
            {
                #if defined (__WIN32__)
                    asThreadViewerLayerManagerZoomIn *thread = new asThreadViewerLayerManagerZoomIn(m_ViewerLayerManagerLeft, &m_CritSectionViewerLayerManager, fittedRect);
                    ThreadsManager().AddThread(thread);
                #else
                    m_ViewerLayerManagerLeft->Zoom(fittedRect);
                #endif
            }
            if (m_DisplayPanelRight)
            {
                #if defined (__WIN32__)
                    asThreadViewerLayerManagerZoomIn *thread = new asThreadViewerLayerManagerZoomIn(m_ViewerLayerManagerRight, &m_CritSectionViewerLayerManager, fittedRect);
                    ThreadsManager().AddThread(thread);
                #else
                    m_ViewerLayerManagerRight->Zoom(fittedRect);
                #endif
            }
		}
	}
	else if(msg->m_EvtType == vrEVT_TOOL_ZOOMOUT)
    {
		// getting rectangle
		vrCoordinate * coord = msg->m_ParentManager->GetDisplay()->GetCoordinate();
		wxASSERT(coord);

		// get real rectangle
		vrRealRect realRect;
		bool success = coord->ConvertFromPixels(msg->m_Rect, realRect);
		wxASSERT(success == true);

		// Get fitted rectangle
		vrRealRect fittedRect = coord->GetRectFitted(realRect);
		wxASSERT(fittedRect.IsOk());

		if (m_SyncroTool == false)
		{
		    #if defined (__WIN32__)
                asThreadViewerLayerManagerZoomOut *thread = new asThreadViewerLayerManagerZoomOut(msg->m_ParentManager, &m_CritSectionViewerLayerManager, fittedRect);
                ThreadsManager().AddThread(thread);
            #else
                msg->m_ParentManager->ZoomOut(fittedRect);
            #endif
		}
		else
		{
		    if (m_DisplayPanelLeft)
            {
                #if defined (__WIN32__)
                    asThreadViewerLayerManagerZoomOut *thread = new asThreadViewerLayerManagerZoomOut(m_ViewerLayerManagerLeft, &m_CritSectionViewerLayerManager, fittedRect);
                    ThreadsManager().AddThread(thread);
                #else
                    m_ViewerLayerManagerLeft->ZoomOut(fittedRect);
                #endif
            }
            if (m_DisplayPanelRight)
            {
                #if defined (__WIN32__)
                    asThreadViewerLayerManagerZoomOut *thread = new asThreadViewerLayerManagerZoomOut(m_ViewerLayerManagerRight, &m_CritSectionViewerLayerManager, fittedRect);
                    ThreadsManager().AddThread(thread);
                #else
                    m_ViewerLayerManagerRight->ZoomOut(fittedRect);
                #endif
            }
		}
	}
	else if (msg->m_EvtType == vrEVT_TOOL_PAN)
	{
		vrCoordinate * coord = msg->m_ParentManager->GetDisplay()->GetCoordinate();
		wxASSERT(coord);

		wxPoint movedPos = msg->m_Position;
		wxPoint2DDouble myMovedRealPt;
		if (coord->ConvertFromPixels(movedPos, myMovedRealPt)==false)
        {
			wxLogError("Error converting point : %d, %d to real coordinate",
					   movedPos.x, movedPos.y);
			wxDELETE(msg);
			return;
		}

		vrRealRect actExtent = coord->GetExtent();
		actExtent.MoveLeftTopTo(myMovedRealPt);

		if (m_SyncroTool == false)
        {
			coord->SetExtent(actExtent);
			msg->m_ParentManager->Reload();
			ReloadViewerLayerManagerLeft();
			ReloadViewerLayerManagerRight();
		}
		else
        {
            if (m_DisplayPanelLeft)
            {
                m_ViewerLayerManagerLeft->GetDisplay()->GetCoordinate()->SetExtent(actExtent);
                ReloadViewerLayerManagerLeft();
            }
            if (m_DisplayPanelRight)
            {
                m_ViewerLayerManagerRight->GetDisplay()->GetCoordinate()->SetExtent(actExtent);
                ReloadViewerLayerManagerRight();
            }
		}

	}
	else if (msg->m_EvtType == vrEVT_TOOL_SIGHT)
	{
		vrViewerLayerManager * invertedMgr = m_ViewerLayerManagerLeft;
		if (invertedMgr == msg->m_ParentManager)
        {
			invertedMgr = m_ViewerLayerManagerRight;
		}

        {
            wxClientDC dc (invertedMgr->GetDisplay());
            wxDCOverlay overlaydc (m_Overlay, &dc);
            overlaydc.Clear();
        }

        m_Overlay.Reset();

		if (msg->m_Position != wxDefaultPosition)
        {
			wxClientDC dc (invertedMgr->GetDisplay());
			wxDCOverlay overlaydc (m_Overlay, &dc);
			overlaydc.Clear();
			dc.SetPen(*wxGREEN_PEN);
			dc.CrossHair(msg->m_Position);
		}
	}
	else
    {
		wxLogError("Operation not supported now");
	}

	wxDELETE(msg);
}

void asFramePredictors::UpdateLayers()
{
    // Check that elements are selected
    if ( (m_SelectedForecast==-1) || (m_SelectedTargetDate==-1) || (m_SelectedAnalogDate==-1) ) return;
    if ( m_SelectedForecast >= m_ForecastManager->GetModelsNb() ) return;

    // Get dates
    asResultsAnalogsForecast* forecast = m_ForecastManager->GetCurrentForecast(m_SelectedForecast);
    Array1DFloat targetDates = forecast->GetTargetDates();
    double targetDate = targetDates[m_SelectedTargetDate];
    Array1DFloat analogDates = forecast->GetAnalogsDates(m_SelectedTargetDate);
    double analogDate = analogDates[m_SelectedAnalogDate];

    m_PredictorsViewer->Redraw(targetDate, analogDate);
}

void asFramePredictors::ReloadViewerLayerManagerLeft( )
{
    #if defined (__WIN32__)
        asThreadViewerLayerManagerReload *thread = new asThreadViewerLayerManagerReload(m_ViewerLayerManagerLeft, &m_CritSectionViewerLayerManager);
        ThreadsManager().AddThread(thread);
    #else
        m_ViewerLayerManagerLeft->Reload();
    #endif
}

void asFramePredictors::ReloadViewerLayerManagerRight( )
{
    #if defined (__WIN32__)
        asThreadViewerLayerManagerReload *thread = new asThreadViewerLayerManagerReload(m_ViewerLayerManagerRight, &m_CritSectionViewerLayerManager);
        ThreadsManager().AddThread(thread);
    #else
        m_ViewerLayerManagerRight->Reload();
    #endif
}

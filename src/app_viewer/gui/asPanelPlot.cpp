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
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#include "asPanelPlot.h"

#include "asIncludes.h"
#include "wx/plotctrl/plotctrl.h"
#include "wx/plotctrl/plotprnt.h"
#include <wx/dcsvg.h>

extern wxString wxPlotCtrl_GetEventName(wxEventType eventType);


BEGIN_EVENT_TABLE(asPanelPlot, wxPanel)
    EVT_PLOTCTRL_ADD_CURVE          (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_DELETING_CURVE     (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_DELETED_CURVE      (wxID_ANY, asPanelPlot::OnPlotCtrl)

    EVT_PLOTCTRL_CURVE_SEL_CHANGING (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_CURVE_SEL_CHANGED  (wxID_ANY, asPanelPlot::OnPlotCtrl)

    EVT_PLOTCTRL_MOUSE_MOTION       (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_CLICKED            (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_DOUBLECLICKED      (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_POINT_CLICKED      (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_POINT_DOUBLECLICKED(wxID_ANY, asPanelPlot::OnPlotCtrl)

    EVT_PLOTCTRL_AREA_SEL_CREATING  (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_AREA_SEL_CHANGING  (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_AREA_SEL_CREATED   (wxID_ANY, asPanelPlot::OnPlotCtrl)

    EVT_PLOTCTRL_VIEW_CHANGING      (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_VIEW_CHANGED       (wxID_ANY, asPanelPlot::OnPlotCtrl)

    EVT_PLOTCTRL_CURSOR_CHANGING    (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_CURSOR_CHANGED     (wxID_ANY, asPanelPlot::OnPlotCtrl)

    EVT_PLOTCTRL_ERROR              (wxID_ANY, asPanelPlot::OnPlotCtrl)

    EVT_PLOTCTRL_BEGIN_TITLE_EDIT   (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_END_TITLE_EDIT     (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_BEGIN_X_LABEL_EDIT (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_END_X_LABEL_EDIT   (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_BEGIN_Y_LABEL_EDIT (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_END_Y_LABEL_EDIT   (wxID_ANY, asPanelPlot::OnPlotCtrl)

    EVT_PLOTCTRL_MOUSE_FUNC_CHANGING (wxID_ANY, asPanelPlot::OnPlotCtrl)
    EVT_PLOTCTRL_MOUSE_FUNC_CHANGED  (wxID_ANY, asPanelPlot::OnPlotCtrl)

END_EVENT_TABLE()


asPanelPlot::asPanelPlot( wxWindow* parent )
:
asPanelPlotVirtual( parent )
{
    m_PlotCtrl->SetScrollOnThumbRelease(false);
    m_PlotCtrl->SetCrossHairCursor(false);
    m_PlotCtrl->SetDrawSymbols(false);
    m_PlotCtrl->SetDrawLines(true);
    m_PlotCtrl->SetDrawSpline(false);
    m_PlotCtrl->SetDrawGrid(true);
    m_PlotCtrl->SetShowXAxis(true);
    m_PlotCtrl->SetShowYAxis(true);
    m_PlotCtrl->SetShowXAxisLabel(true);
    m_PlotCtrl->SetShowYAxisLabel(true);
    m_PlotCtrl->SetShowPlotTitle(false);
    m_PlotCtrl->SetShowKey(true);
}

wxPlotCtrl* asPanelPlot::GetPlotCtrl()
{
    wxASSERT(m_PlotCtrl);
    return m_PlotCtrl;
}

void asPanelPlot::OnPlotCtrl(wxPlotCtrlEvent& event)
{
    // Check that the pointer is set
    if (!m_PlotCtrl) return;

    // Get event type
    wxEventType eventType = event.GetEventType();

    // Process according to event
    if (eventType == wxEVT_PLOTCTRL_ADD_CURVE)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_DELETING_CURVE)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_DELETED_CURVE)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_CURVE_SEL_CHANGING)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_CURVE_SEL_CHANGED)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_MOUSE_MOTION)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_CLICKED)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_DOUBLECLICKED)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_POINT_CLICKED)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_POINT_DOUBLECLICKED)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_AREA_SEL_CREATING)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_AREA_SEL_CHANGING)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_AREA_SEL_CREATED)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_VIEW_CHANGING)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_VIEW_CHANGED)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_CURSOR_CHANGING)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_CURSOR_CHANGED)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_ERROR)
	{
	    asLogError(event.GetString());
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_BEGIN_TITLE_EDIT)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_END_TITLE_EDIT)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_BEGIN_X_LABEL_EDIT)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_END_X_LABEL_EDIT)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_BEGIN_Y_LABEL_EDIT)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_END_Y_LABEL_EDIT)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_MOUSE_FUNC_CHANGING)
	{
		event.Skip();
	}
	else if (eventType == wxEVT_PLOTCTRL_MOUSE_FUNC_CHANGED)
	{
		event.Skip();
	}

	event.Skip();
}

void asPanelPlot::Print()
{
    wxPlotPrintout plotPrint(m_PlotCtrl, wxT("AtmoSwing Printout"));

    plotPrint.ShowPrintDialog();
}

void asPanelPlot::PrintPreview()
{
    wxPlotPrintout plotPrint(m_PlotCtrl, wxT("AtmoSwing Printout"));

    plotPrint.ShowPrintPreviewDialog(wxT("AtmoSwing Print Preview"));
}

void asPanelPlot::ExportSVG()
{
    wxFileDialog dialog(this, wxT("Save SVG file as"), wxEmptyString, "AtmoSwing_timeseries",
        wxT("SVG vector picture files (*.svg)|*.svg"),
        wxFD_SAVE|wxFD_OVERWRITE_PROMPT);

    if (dialog.ShowModal() == wxID_OK)
    {
        double dpi = 72;

        wxSVGFileDC svgDC (dialog.GetPath(), 600, 400, dpi);
        wxRect rect(0, 0, 600, 400);

        GetPlotCtrl()->DrawWholePlot( &svgDC, rect, dpi );

        if(!svgDC.IsOk())
        {
            asLogError(_("The svg DC is not OK."));
            return;
        }
    }

    return;
}

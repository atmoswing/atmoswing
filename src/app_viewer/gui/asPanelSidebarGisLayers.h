/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#ifndef __asPanelSidebarGisLayers__
#define __asPanelSidebarGisLayers__

#include "asPanelSidebar.h"

#include "asIncludes.h"
#include "vroomgis.h"

/** Implementing asPanelSidebarGisLayers */
class asPanelSidebarGisLayers : public asPanelSidebar
{
public:
    /** Constructor */
    asPanelSidebarGisLayers( wxWindow* parent, wxWindowID id = wxID_ANY, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxTAB_TRAVERSAL);
    ~asPanelSidebarGisLayers();

    vrViewerTOCList* GetTocCtrl()
    {
        return m_TocCtrl;
    }

private:
    // vroomgis
	vrViewerTOCList *m_TocCtrl;

	//void OnPaint( wxCommandEvent& event );
};

#endif // __asPanelSidebarGisLayers__

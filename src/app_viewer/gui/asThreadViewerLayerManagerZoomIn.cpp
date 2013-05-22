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
 
#include "asThreadViewerLayerManagerZoomIn.h"

#include <asTimeArray.h>
#include <asThreadsManager.h>


asThreadViewerLayerManagerZoomIn::asThreadViewerLayerManagerZoomIn(vrViewerLayerManager *viewerLayerManager, wxCriticalSection *critSectionViewerLayerManager, const vrRealRect &fittedRect)
:
asThread()
{
    m_Status = Initializing;

    m_ViewerLayerManager = viewerLayerManager;
    m_CritSectionViewerLayerManager = critSectionViewerLayerManager;
    m_Rect = fittedRect;

    wxASSERT(m_ViewerLayerManager);

    m_Status = Waiting;
}

asThreadViewerLayerManagerZoomIn::~asThreadViewerLayerManagerZoomIn()
{

}

wxThread::ExitCode asThreadViewerLayerManagerZoomIn::Entry()
{
    m_Status = Working;

    m_CritSectionViewerLayerManager->Enter();
    m_ViewerLayerManager->Zoom(m_Rect);
    m_CritSectionViewerLayerManager->Leave();

    return 0;
}

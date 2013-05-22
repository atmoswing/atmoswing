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
 
#include "asThreadViewerLayerManagerZoomOut.h"

#include <asTimeArray.h>
#include <asThreadsManager.h>


asThreadViewerLayerManagerZoomOut::asThreadViewerLayerManagerZoomOut(vrViewerLayerManager *viewerLayerManager, wxCriticalSection *critSectionViewerLayerManager, const vrRealRect &fittedRect)
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

asThreadViewerLayerManagerZoomOut::~asThreadViewerLayerManagerZoomOut()
{

}

wxThread::ExitCode asThreadViewerLayerManagerZoomOut::Entry()
{
    m_Status = Working;

    m_CritSectionViewerLayerManager->Enter();
    m_ViewerLayerManager->ZoomOut(m_Rect);
    m_CritSectionViewerLayerManager->Leave();

    return 0;
}

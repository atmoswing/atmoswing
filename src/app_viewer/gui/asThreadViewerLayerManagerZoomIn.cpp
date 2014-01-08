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

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
 
#include "asThreadViewerLayerManagerZoomOut.h"

#include <asTimeArray.h>
#include <asThreadsManager.h>


asThreadViewerLayerManagerZoomOut::asThreadViewerLayerManagerZoomOut(vrViewerLayerManager *viewerLayerManager, wxCriticalSection *critSectionViewerLayerManager, const vrRealRect &fittedRect)
:
asThread()
{
    m_status = Initializing;

    m_viewerLayerManager = viewerLayerManager;
    m_critSectionViewerLayerManager = critSectionViewerLayerManager;
    m_rect = fittedRect;

    wxASSERT(m_viewerLayerManager);

    m_status = Waiting;
}

asThreadViewerLayerManagerZoomOut::~asThreadViewerLayerManagerZoomOut()
{

}

wxThread::ExitCode asThreadViewerLayerManagerZoomOut::Entry()
{
    m_status = Working;

    m_critSectionViewerLayerManager->Enter();
    m_viewerLayerManager->ZoomOut(m_rect);
    m_critSectionViewerLayerManager->Leave();

    return 0;
}

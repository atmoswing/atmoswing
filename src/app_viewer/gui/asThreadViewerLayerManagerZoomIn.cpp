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
 */

#include "asThreadViewerLayerManagerZoomIn.h"

asThreadViewerLayerManagerZoomIn::asThreadViewerLayerManagerZoomIn(vrViewerLayerManager *viewerLayerManager,
                                                                   wxCriticalSection *critSectionViewerLayerManager,
                                                                   const vrRealRect &fittedRect)
    : asThread(),
      m_viewerLayerManager(viewerLayerManager),
      m_critSectionViewerLayerManager(critSectionViewerLayerManager),
      m_rect(fittedRect) {
  wxASSERT(m_viewerLayerManager);
}

wxThread::ExitCode asThreadViewerLayerManagerZoomIn::Entry() {
  m_critSectionViewerLayerManager->Enter();
  m_viewerLayerManager->Zoom(m_rect);
  m_critSectionViewerLayerManager->Leave();

  return (wxThread::ExitCode)0;
}

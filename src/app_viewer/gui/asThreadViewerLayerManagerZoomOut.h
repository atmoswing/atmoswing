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
 
#ifndef asThreadViewerLayerManagerZoomOut_H
#define asThreadViewerLayerManagerZoomOut_H

#include <asThread.h>
#include <asIncludes.h>
#include "vroomgis.h"

class asThreadViewerLayerManagerZoomOut: public asThread
{
public:
    /** Default constructor */
    asThreadViewerLayerManagerZoomOut(vrViewerLayerManager *viewerLayerManager, wxCriticalSection *critSectionViewerLayerManager, const vrRealRect &fittedRect);
    /** Default destructor */
    virtual ~asThreadViewerLayerManagerZoomOut();

    virtual ExitCode Entry();


protected:
private:
    vrViewerLayerManager *m_ViewerLayerManager;
    wxCriticalSection *m_CritSectionViewerLayerManager;
    vrRealRect m_Rect;

};

#endif // asThreadViewerLayerManagerZoomOut_H

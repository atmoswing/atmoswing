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
 
#ifndef asThreadViewerLayerManagerZoomIn_H
#define asThreadViewerLayerManagerZoomIn_H

#include <asThread.h>
#include <asIncludes.h>
#include "vroomgis.h"

class asThreadViewerLayerManagerZoomIn: public asThread
{
public:
    /** Default constructor */
    asThreadViewerLayerManagerZoomIn(vrViewerLayerManager *viewerLayerManager, wxCriticalSection *critSectionViewerLayerManager, const vrRealRect &fittedRect);
    /** Default destructor */
    virtual ~asThreadViewerLayerManagerZoomIn();

    virtual ExitCode Entry();


protected:
private:
    vrViewerLayerManager *m_ViewerLayerManager;
    wxCriticalSection *m_CritSectionViewerLayerManager;
    vrRealRect m_Rect;

};

#endif // asThreadViewerLayerManagerZoomIn_H

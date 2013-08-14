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
 
#ifndef asThreadViewerLayerManagerReload_H
#define asThreadViewerLayerManagerReload_H

#include <asThread.h>
#include <asIncludes.h>
#include "vroomgis.h"

class asThreadViewerLayerManagerReload: public asThread
{
public:
    /** Default constructor */
    asThreadViewerLayerManagerReload(vrViewerLayerManager *viewerLayerManager, wxCriticalSection *critSectionViewerLayerManager);
    /** Default destructor */
    virtual ~asThreadViewerLayerManagerReload();

    virtual ExitCode Entry();


protected:
private:
    vrViewerLayerManager *m_ViewerLayerManager;
    wxCriticalSection *m_CritSectionViewerLayerManager;

};

#endif // asThreadViewerLayerManagerReload_H

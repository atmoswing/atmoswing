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
 
#ifndef _VRLAYERVECTORSFCSTDOTS_H_
#define _VRLAYERVECTORSFCSTDOTS_H_

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
// Include wxWidgets' headers
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif


#include "asIncludes.h"
#include "vrlayervector.h"

class vrRender;
class vrLabel;

//For dealing with GIS data stored into Fcst projects.
class vrLayerVectorFcstDots : public vrLayerVectorOGR
{
public:
    vrLayerVectorFcstDots();
    virtual ~vrLayerVectorFcstDots();

    virtual long AddFeature(OGRGeometry * geometry, void * data = NULL);

    void SetMaxValue(double val)
    {
        if (val<0.1)
        {
            asLogWarning(_("The given maximum value for the vrLayerVectorFcstDots class was too small, so it has been increased."));
            val = 0.1;
        }
        m_valueMax = val;
    }

    double GetMaxValue()
    {
        return m_valueMax;
    }

protected:
    double m_valueMax;

    virtual void _DrawPoint(wxDC * dc, OGRFeature * feature, OGRGeometry * geometry, const wxRect2DDouble & coord, const vrRender * render,  vrLabel * label, double pxsize);

    void _CreatePath(wxGraphicsPath & path, const wxPoint & center);

    void _Paint(wxGraphicsContext * gdc, wxGraphicsPath & path, double value);

    void _AddLabel(wxGraphicsContext * gdc, const wxPoint & center, double value);
};
#endif


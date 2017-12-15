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
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asGeo.h"

asGeo::asGeo(GridType type)
        : m_gridType(type),
          m_axisXmin(0),
          m_axisXmax(360),
          m_axisYmin(-90),
          m_axisYmax(90)
{
    // We always consider WGS84 for the predictors
}

asGeo::~asGeo()
{
    //dtor
}

bool asGeo::CheckPoint(Coo &point, int changesAllowed)
{
    // We always consider WGS84 for the predictors
    if (point.y < m_axisYmin) {
        if (changesAllowed == asEDIT_ALLOWED) {
            point.y = m_axisYmin + (m_axisYmin - point.y);
            point.x = point.x + 180;
        }
        return false;
    }
    if (point.y > m_axisYmax) {
        if (changesAllowed == asEDIT_ALLOWED) {
            point.y = m_axisYmax + (m_axisYmax - point.y);
            point.x = point.x + 180;
        }
        return false;
    }
    if (point.x < m_axisXmin) {
        if (changesAllowed == asEDIT_ALLOWED) {
            point.x += m_axisXmax;
        }
        return false;
    }
    if (point.x > m_axisXmax) {
        if (changesAllowed == asEDIT_ALLOWED) {
            point.x -= m_axisXmax;
        }
        return false;
    }

    return true;
}

wxString asGeo::GetGridTypeString() const
{
    switch (m_gridType) {
        case (Regular):
            return "Regular";
        case (GaussianT62):
            return "GaussianT62";
        case (GaussianT382):
            return "GaussianT382";
        default:
            return "Not found";
    }
}
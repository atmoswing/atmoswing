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

asGeo::asGeo()
{
    InitBounds();
}

asGeo::~asGeo()
{
    //dtor
}

void asGeo::InitBounds()
{
    // We always consider WGS84 for the predictors
    m_axisXmin = 0;
    m_axisXmax = 360;
    m_axisYmin = -90;
    m_axisYmax = 90;
}

bool asGeo::CheckPoint(Coo &Point, int ChangesAllowed)
{
    // We always consider WGS84 for the predictors
    if (Point.y < m_axisYmin) {
        if (ChangesAllowed == asEDIT_ALLOWED) {
            Point.y = m_axisYmin + (m_axisYmin - Point.y);
            Point.x = Point.x + 180;
        }
        return false;
    }
    if (Point.y > m_axisYmax) {
        if (ChangesAllowed == asEDIT_ALLOWED) {
            Point.y = m_axisYmax + (m_axisYmax - Point.y);
            Point.x = Point.x + 180;
        }
        return false;
    }
    if (Point.x < m_axisXmin) {
        if (ChangesAllowed == asEDIT_ALLOWED) {
            Point.x += m_axisXmax;
        }
        return false;
    }
    if (Point.x > m_axisXmax) {
        if (ChangesAllowed == asEDIT_ALLOWED) {
            Point.x -= m_axisXmax;
        }
        return false;
    }

    return true;
}

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
    m_AxisXmin = 0;
    m_AxisXmax = 360;
    m_AxisYmin = -90;
    m_AxisYmax = 90;
}

bool asGeo::CheckPoint(Coo &Point, int ChangesAllowed)
{
    // We always consider WGS84 for the predictors
    if(Point.y<m_AxisYmin)
    {
        if (ChangesAllowed == asEDIT_ALLOWED)
        {
            Point.y = m_AxisYmin + (m_AxisYmin - Point.y);
            Point.x = Point.x + 180;
        }
        return false;
    }
    if(Point.y>m_AxisYmax)
    {
        if (ChangesAllowed == asEDIT_ALLOWED)
        {
            Point.y = m_AxisYmax + (m_AxisYmax - Point.y);
            Point.x = Point.x + 180;
        }
        return false;
    }
    if(Point.x<m_AxisXmin)
    {
        if (ChangesAllowed == asEDIT_ALLOWED)
        {
            Point.x += m_AxisXmax;
        }
        return false;
    }
    if(Point.x>m_AxisXmax)
    {
        if (ChangesAllowed == asEDIT_ALLOWED)
        {
            Point.x -= m_AxisXmax;
        }
        return false;
    }

    return true;
}

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

#include "asAreaCompRegGrid.h"

asAreaCompRegGrid::asAreaCompRegGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                                     double xStep, double yStep, int flatAllowed)
        : asAreaCompGrid(cornerUL, cornerUR, cornerLL, cornerLR, flatAllowed),
          m_xStep(xStep),
          m_yStep(yStep)
{
    m_isRegular = true;
}

asAreaCompRegGrid::asAreaCompRegGrid(double xMin, double xWidth, double xStep, double yMin, double yWidth, double yStep,
                                     int flatAllowed)
        : asAreaCompGrid(xMin, xWidth, yMin, yWidth, flatAllowed),
          m_xStep(xStep),
          m_yStep(yStep)
{
    m_isRegular = true;
}

bool asAreaCompRegGrid::GridsOverlay(asAreaCompGrid *otherArea) const
{
    if (!otherArea->IsRegular())
        return false;

    auto *otherAreaRegular(dynamic_cast<asAreaCompRegGrid *>(otherArea));

    if (!otherAreaRegular)
        return false;

    if (GetXstep() != otherAreaRegular->GetXstep())
        return false;

    if (GetYstep() != otherAreaRegular->GetYstep())
        return false;

    return true;
}
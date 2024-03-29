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
 * Portions Copyright 2018 Pascal Horton, University of Bern.
 */

#include "asAreaGridGeneric.h"

#include "asTypeDefs.h"

asAreaGridGeneric::asAreaGridGeneric(const Coo& cornerUL, const Coo& cornerUR, const Coo& cornerLL, const Coo& cornerLR,
                                     int flatAllowed, bool isLatLon)
    : asAreaGrid(cornerUL, cornerUR, cornerLL, cornerLR, flatAllowed, isLatLon) {}

asAreaGridGeneric::asAreaGridGeneric(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed,
                                     bool isLatLon)
    : asAreaGrid(xMin, xWidth, yMin, yWidth, flatAllowed, isLatLon) {}

asAreaGridGeneric::asAreaGridGeneric(double xMin, int xPtsNb, double yMin, int yPtsNb, int flatAllowed, bool isLatLon)
    : asAreaGrid(xMin, 0, yMin, 0, flatAllowed, isLatLon) {
    m_xPtsNb = xPtsNb;
    m_yPtsNb = yPtsNb;
}

bool asAreaGridGeneric::GridsOverlay(asAreaGrid* otherArea) const {
    return false;
}

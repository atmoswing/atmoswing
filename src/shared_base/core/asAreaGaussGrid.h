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

#ifndef asAreaGaussianGrid_H
#define asAreaGaussianGrid_H

#include <asIncludes.h>
#include <asArea.h>

class asAreaGaussGrid
        : public asArea
{
public:
    asAreaGaussGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR, GridType type,
                    float level = asNONE, int flatAllowed = asFLAT_ALLOWED);

    asAreaGaussGrid(double xMin, int xPtsNb, double yMin, int yPtsNb, GridType type, float level = asNONE,
                    int flatAllowed = asFLAT_ALLOWED);

    ~asAreaGaussGrid() override = default;

    static void BuildLonAxis(a1d &axis, const GridType &type);

    static void BuildLatAxis(a1d &axis, const GridType &type);

    int GetXaxisPtsnb() const;

    int GetYaxisPtsnb() const;

    a1d GetXaxis();

    a1d GetYaxis();

protected:

private:
    a1d m_fullAxisX;
    a1d m_fullAxisY;

    bool IsOnGrid(const Coo &point) const;

    bool IsOnGrid(double xCoord, double yCoord) const;
};

#endif // asAreaGaussianGrid_H

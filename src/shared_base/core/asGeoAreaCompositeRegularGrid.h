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

#ifndef asGeoAreaCompositeRegularGrid_H
#define asGeoAreaCompositeRegularGrid_H

#include <asIncludes.h>
#include <asGeoAreaCompositeGrid.h>

class asGeoAreaCompositeRegularGrid
        : public asGeoAreaCompositeGrid
{
public:
    asGeoAreaCompositeRegularGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                                  double xStep, double yStep, float level = asNONE, float height = asNONE,
                                  int flatAllowed = asFLAT_FORBIDDEN);

    asGeoAreaCompositeRegularGrid(double xMin, double xWidth, double xStep, double yMin, double yWidth, double yStep,
                                  float level = asNONE, float height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    virtual ~asGeoAreaCompositeRegularGrid();

    bool GridsOverlay(asGeoAreaCompositeGrid *otherarea) const;

    double GetXstep() const
    {
        return m_xStep;
    }

    double GetYstep() const
    {
        return m_yStep;
    }

    a1d GetXaxisComposite(int compositeNb);

    a1d GetYaxisComposite(int compositeNb);

    int GetXaxisCompositePtsnb(int compositeNb);

    int GetYaxisCompositePtsnb(int compositeNb);

    double GetXaxisCompositeWidth(int compositeNb) const;

    double GetYaxisCompositeWidth(int compositeNb) const;

    double GetXaxisCompositeStart(int compositeNb) const;

    double GetYaxisCompositeStart(int compositeNb) const;

    double GetXaxisCompositeEnd(int compositeNb) const;

    double GetYaxisCompositeEnd(int compositeNb) const;

protected:

private:
    double m_xStep;
    double m_yStep;

    bool IsOnGrid(double step) const;

    bool IsOnGrid(double stepX, double stepY) const;
};

#endif // asGeoAreaCompositeRegularGrid_H

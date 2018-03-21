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

#ifndef asAreaCompositeGaussianGrid_H
#define asAreaCompositeGaussianGrid_H

#include <asIncludes.h>
#include <asAreaCompGrid.h>
#include <asAreaGaussGrid.h>

class asAreaCompGaussGrid
        : public asAreaCompGrid
{
public:
    asAreaCompGaussGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                        GridType type = GaussianT62, float level = asNONE, float height = asNONE,
                        int flatAllowed = asFLAT_FORBIDDEN);

    asAreaCompGaussGrid(double xMin, int xPtsNb, double yMin, int yPtsNb, GridType type = GaussianT382,
                        float level = asNONE, float height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    ~asAreaCompGaussGrid() override = default;

    bool GridsOverlay(asAreaCompGrid *otherarea) const override;

    double GetXstep() const override
    {
        return 0.0;
    }

    double GetYstep() const override
    {
        return 0.0;
    }

    a1d GetXaxisComposite(int compositeNb) override;

    a1d GetYaxisComposite(int compositeNb) override;

    int GetXaxisCompositePtsnb(int compositeNb) override;

    int GetYaxisCompositePtsnb(int compositeNb) override;

    double GetXaxisCompositeWidth(int compositeNb) const override;

    double GetYaxisCompositeWidth(int compositeNb) const override;

    double GetXaxisCompositeStart(int compositeNb) const override;

    double GetYaxisCompositeStart(int compositeNb) const override;

    double GetXaxisCompositeEnd(int compositeNb) const override;

    double GetYaxisCompositeEnd(int compositeNb) const override;

protected:

private:
    a1d m_fullAxisX;
    a1d m_fullAxisY;

    bool IsOnGrid(const Coo &point) const;

    bool IsOnGrid(double xCoord, double yCoord) const;
};

#endif // asAreaCompositeGaussianGrid_H

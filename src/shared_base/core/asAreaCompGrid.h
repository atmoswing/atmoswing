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

#ifndef asAreaCompositeGrid_H
#define asAreaCompositeGrid_H

#include <asIncludes.h>
#include <asAreaComp.h>

class asParameters;

class asAreaCompGrid
        : public asAreaComp
{
public:

    static asAreaCompGrid * GetInstance(const asParameters *params, int iStep, int iPtor);

    asAreaCompGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                   int flatAllowed = asFLAT_FORBIDDEN);

    asAreaCompGrid(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed = asFLAT_FORBIDDEN);

    asAreaCompGrid();

    wxString GetGridTypeString() const;

    bool InitializeAxes(const a1f &lons, const a1f &lats);

    virtual bool GridsOverlay(asAreaCompGrid *otherArea) const = 0;

    virtual a1d GetXaxisComposite(int compositeNb);

    virtual a1d GetYaxisComposite(int compositeNb);

    virtual int GetXaxisCompositePtsnb(int compositeNb);

    virtual int GetYaxisCompositePtsnb(int compositeNb);

    double GetXaxisCompositeStart(int compositeNb) const;

    double GetYaxisCompositeStart(int compositeNb) const;

    double GetXaxisCompositeEnd(int compositeNb) const;

    double GetYaxisCompositeEnd(int compositeNb) const;

    virtual double GetXstep() const = 0;

    virtual double GetYstep() const = 0;

    bool IsRegular() const
    {
        return m_isRegular;
    }


protected:
    bool m_isRegular;
    bool m_isInitialized;
    std::vector<a1d> m_compositeXaxes;
    std::vector<a1d> m_compositeYaxes;
    int m_xPtsNb;
    int m_yPtsNb;

private:

};

#endif // asAreaCompositeGrid_H

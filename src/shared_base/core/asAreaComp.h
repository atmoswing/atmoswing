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

#ifndef ASAREACOMPOSITE_H
#define ASAREACOMPOSITE_H

#include <asIncludes.h>
#include <asArea.h>

class asAreaComp
        : public asArea
{
public:
    asAreaComp(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
               int flatAllowed = asFLAT_FORBIDDEN, bool isLatLon = true);

    asAreaComp(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed = asFLAT_FORBIDDEN,
               bool isLatLon = true);

    asAreaComp();

    ~asAreaComp() override = default;

    double GetXmin() const override;

    double GetXmax() const override;

    double GetYmin() const override;

    double GetYmax() const override;

    int GetNbComposites() const
    {
        return (int) m_composites.size();
    }

    asArea GetComposite(int id) const
    {
        wxASSERT(m_composites.size() > id);

        return m_composites[id];
    }

protected:
    std::vector<asArea> m_composites;

    void Init() override;

    void CreateComposites();

    bool DoCheckPoints();

    bool CheckConsistency();

private:
};

#endif

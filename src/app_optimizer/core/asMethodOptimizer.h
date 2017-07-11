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

#ifndef ASMETHODOPTIMIZER_H
#define ASMETHODOPTIMIZER_H

#include "asIncludes.h"
#include <asMethodCalibrator.h>
#include <asParametersOptimization.h>


class asMethodOptimizer
        : public asMethodCalibrator
{
public:
    asMethodOptimizer();

    virtual ~asMethodOptimizer();

    virtual bool Manager() = 0;

protected:
    bool m_isOver;
    bool m_skipNext;
    int m_optimizerStage;
    int m_paramsNb;
    int m_iterator;

    virtual bool Calibrate(asParametersCalibration &params)
    {
        wxLogError(_("asMethodOptimizer do optimize, not calibrate..."));
        return false;
    }

    bool SaveDetails(asParametersOptimization &params);

    bool Validate(asParametersOptimization &params);

    void IncrementIterator()
    {
        m_iterator++;
    }

    bool IsOver() const
    {
        return m_isOver;
    }

    bool SkipNext() const
    {
        return m_skipNext;
    }

private:

};

#endif // ASMETHODOPTIMIZER_H

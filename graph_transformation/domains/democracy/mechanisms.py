# In domains/democracy/mechanisms.py
def create_pld_mechanism(config) -> Transform:
    """Creates a Predictive Liquid Democracy mechanism."""
    # Component transformations
    information_sharing = create_information_sharing_transform(config)
    delegation = create_delegation_transform(config)
    voting_power_calculation = create_voting_power_calculation_transform(config)
    prediction_market = create_prediction_market_transform(config)
    liquid_voting = create_liquid_voting_transform(config)
    resource_application = create_resource_application_transform(config)
    
    # Attach properties to transformations
    voting_power_calculation = attach_properties(
        voting_power_calculation, 
        {voting_power_conservation}
    )
    
    delegation = attach_properties(
        delegation, 
        {delegation_acyclicity}
    )
    
    # Compose the mechanism
    mechanism = sequential(
        information_sharing,
        delegation,
        voting_power_calculation,
        prediction_market,
        liquid_voting,
        resource_application
    )
    
    # The properties of the composition will be the intersection
    # of properties preserved by each component transformation
    return mechanism